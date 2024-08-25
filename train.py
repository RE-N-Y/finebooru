import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import lpips
import wandb
import deeplake

from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from modeling.base import VQVAE, CVQVAE
from modeling.discrminator import Discriminator
from augmentation import daugment

import math
import click
from functools import partial
from tqdm import tqdm
from pathlib import Path

identity = lambda t : t
vjp = partial(torch.autograd.grad, create_graph=True, retain_graph=True, only_inputs=True)
differentiable = lambda f : f.requires_grad_(True)

def DRLoss(D, reals:Tensor, steps=4, weight=1.):
    y = D(differentiable(reals))
    ones = torch.ones(y.shape, device=y.device, dtype=y.dtype)
    gradients, *_ = vjp(outputs=y, inputs=reals, grad_outputs=ones)
    gradients = gradients.view(len(gradients), -1)
    norm = gradients.norm(2, dim=-1)
    penalty = (norm - 1) ** 2

    loss = .5 * weight * steps * penalty
    return loss.mean()


def DLoss(G, D, reals, augmentation=identity):
    fakes, loss, idxes = G(reals)
    fscores = D(augmentation(fakes))
    rscores = D(augmentation(reals))
    loss = F.softplus(fscores) + F.softplus(-rscores)

    return loss.mean()

def GLoss(G, D, P, reals, augmentation=identity):
    fakes, compress, idxes = G(reals)
    l1 = torch.abs(fakes - reals)
    l2 = torch.square(fakes - reals)
    perceptual = P(reals, fakes)
    adversarial = F.softplus(-D(augmentation(fakes)))

    loss = .1 * perceptual.mean() + \
           .1 * adversarial.mean() + \
            1 * l2.mean()

    return { 
        "loss":loss, 
        "perceptual":perceptual.mean(), 
        "adversarial":adversarial.mean(), 
        "l2":l2.mean(),
        "l1":l1.mean(),
    }

@click.command()
@click.option("--name", default="name", type=str)
@click.option("--dataset", default="hub://activeloop/imagenet-train", type=str)
@click.option("--backbone", default="attention", type=str)
@click.option("--compile", default=False, type=bool)
@click.option("--epochs", default=4, type=int)
@click.option("--size", default=256, type=int)
@click.option("--lr", default=3e-4, type=float)
@click.option("--features", default=768, type=int)
@click.option("--codes", default=8, type=int)
@click.option("--pages", default=16384, type=int)
@click.option("--heads", default=12, type=int)
@click.option("--batch_size", default=64, type=int)
@click.option("--patch", default=16, type=int)
@click.option("--strides", default=16, type=int)
@click.option("--padding", default=0, type=int)
@click.option("--epochs", default=42, type=int)
@click.option("--steps", default=1e+6, type=int)
@click.option("--quantiser", default="vq", type=str)
@click.option("--gradient_accumulation_steps", default=8, type=int)
@click.option("--temperature", default=1, type=float)
@click.option("--log_every_n_steps", default=1024, type=int)
@click.option("--depth", default=12, type=int)
@click.option("--dims", default=[8,8,8,6,5], type=list[int])
@click.option("--beta1", default=0.9, type=float)
@click.option("--beta2", default=0.99, type=float)
@click.option("--wd", default=1e-4, type=float)
@click.option("--save", default=False, type=bool)
@click.option("--push", default=False, type=bool)
@click.option("--useconv", default=False, type=bool)
def main(**config):
    torch.set_float32_matmul_precision('high')
    accelerator = Accelerator(gradient_accumulation_steps=config["gradient_accumulation_steps"], log_with="wandb")
    accelerator.init_trackers("vit", config, init_kwargs={"wandb":{"name":config["name"]}})

    if config["backbone"] == "convolution":
        G = CVQVAE(
            features=config["features"], 
            codes=config["codes"], 
            pages=config["pages"], 
            size=config["size"], 
            quantiser=config["quantiser"], 
            temperature=config["temperature"],
            dims=config["dims"]
        )
    
    elif config["backbone"] in ["attention", "ssd", "bssd", "vssd", "ssdv"]:
        G = VQVAE(
            backbone=config["backbone"], 
            features=config["features"],
            heads=config["heads"],
            codes=config["codes"],
            pages=config["pages"],
            depth=config["depth"],
            patch=config["patch"], 
            size=config["size"],
            strides=config["strides"], 
            padding=config["padding"],
            quantiser=config["quantiser"],
            temperature=config["temperature"],
            dims=config["dims"],
            useconv=config["useconv"]
        )
    else:
        raise ValueError(f"Invalid backbone {config['backbone']}")
    
    P = lpips.LPIPS(net='vgg')
    P = P.eval()

    D = Discriminator(config["size"])
    ds = deeplake.load(config["dataset"])

    tform = T.Compose([
        T.ToTensor(), T.Resize(config["size"], antialias=True),
        T.RandomResizedCrop(config["size"], scale=(0.8, 1), antialias=True),
        T.RandomHorizontalFlip(0.3), T.RandomAdjustSharpness(2,0.3), T.RandomAutocontrast(0.3),
        T.Lambda(lambda x: x.repeat(int( 3 / len(x)), 1, 1)),
        T.ConvertImageDtype(torch.float), T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    dataloader = ds.pytorch(
        tensors=["images"],
        transform={ "images" : tform },
        num_workers=16, batch_size=config["batch_size"],
        decode_method={ "images":"numpy" },
        drop_last=True,
        buffer_size=8192,
        shuffle=True
    )

    if config["compile"]:
        G, P, D = torch.compile(G), torch.compile(P), torch.compile(D)

    Gtx = torch.optim.AdamW(G.parameters(), lr = config["lr"], betas=(config["beta1"], config["beta2"]), weight_decay=config["wd"])
    Dtx = torch.optim.AdamW(D.parameters(), lr = config["lr"], betas=(config["beta1"], config["beta2"]), weight_decay=config["wd"])

    Gscheduler = get_cosine_schedule_with_warmup(Gtx, 4096, config["epochs"] * len(dataloader))
    Dscheduler = get_cosine_schedule_with_warmup(Dtx, 4096, config["epochs"] * len(dataloader))
    size = sum(p.numel() for p in G.parameters() if p.requires_grad) / 1e+6
    accelerator.log({ "parameters" : size })

    G, P, Gtx, Gscheduler, Dscheduler, dataloader = accelerator.prepare(G, P, Gtx, Gscheduler, Dscheduler, dataloader)

    for epoch in tqdm(range(config["epochs"])):
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            losses = { "D" : 0, "G" : 0, "DR" : 0 }
            reals = batch["images"]

            with accelerator.accumulate(D):
                losses["D"] = DLoss(G, D, reals, augmentation=daugment)
                accelerator.backward(losses["D"] + losses["DR"])
                                
                Dtx.step()
                Dscheduler.step()
                Dtx.zero_grad()

            with accelerator.accumulate(G):
                output = GLoss(G, P, reals)
                losses["G"] = output["loss"]
                accelerator.backward(losses["G"])

                Gtx.step()
                Gscheduler.step()
                Gtx.zero_grad()
                

            accelerator.log({ **output, **losses })

            if idx % config["log_every_n_steps"] == 0:
                with torch.no_grad():
                    fakes, _, _ = G(reals)
                    reals, fakes = (reals + 1) / 2, (fakes + 1) / 2
                    reals, fakes = reals.clamp(0,1), fakes.clamp(0,1)
                    accelerator.log({ "samples" : wandb.Image(fakes), "reals" : wandb.Image(reals) })
            
            if idx > config["steps"]:
                accelerator.end_training()
                return
            
    accelerator.end_training()
    
    G_ = accelerator.unwrap_model(G)
    if config["save"]:
        G_.save_pretrained(f"RE-N-Y/{config['name']}")
        if config["push"]:
            G_.push_to_hub(f"RE-N-Y/{config['name']}")



if __name__ == "__main__":
    main()