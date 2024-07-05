import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange, repeat, reduce


import lpips
import wandb
import deeplake

from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from modeling.base import VQVAE, MVQVAE

import math
import click
from functools import partial
from tqdm import tqdm

def GLoss(G, P, reals):
    fakes, compress, idxes = G(reals)
    l1 = torch.abs(fakes - reals)
    l2 = torch.square(fakes - reals)
    perceptual = P(reals, fakes)

    loss =  1 * compress + \
            1 * perceptual.square().mean() + \
           .1 * l1.mean() + \
           .1 * l2.mean()

    return {
        "loss":loss,
        "compress":compress,
        "perceptual":perceptual.mean(),
        "l2":l2.mean(),
        "l1":l1.mean(),
    }

@click.command()
@click.option("--features", default=768, type=int)
@click.option("--backbone", default="attention", type=str)
@click.option("--compile", default=False, type=bool)
@click.option("--size", default=128, type=int)
@click.option("--lr", default=1e-4, type=float)
@click.option("--batch_size", default=64, type=int)
@click.option("--patch", default=16, type=int)
@click.option("--strides", default=16, type=int)
@click.option("--padding", default=0, type=int)
@click.option("--gradient_accumulation_steps", default=4, type=int)
@click.option("--log_every_n_steps", default=1024, type=int)
def main(**config):
    torch.set_float32_matmul_precision('high')
    accelerator = Accelerator(gradient_accumulation_steps=config["gradient_accumulation_steps"], log_with="wandb")
    accelerator.init_trackers("vit", config)

    if config["backbone"] == "attention":
        G = VQVAE(features=config["features"], patch=config["patch"], size=config["size"], strides=config["strides"], padding=config["padding"])
    elif config["backbone"] == "bssd":
        G = MVQVAE(features=config["features"], patch=config["patch"], size=config["size"], strides=config["strides"], padding=config["padding"])
    else:
        raise ValueError(f"Invalid backbone {config['backbone']}")

    P = lpips.LPIPS(net='vgg')
    P = P.eval()

    ds = deeplake.load("hub://reny/animefaces")

    tform = T.Compose([
        T.ToTensor(), T.Resize(config["size"], antialias=True),
        T.RandomResizedCrop(config["size"], scale=(0.8, 1), antialias=True),
        T.RandomHorizontalFlip(0.3), T.RandomAdjustSharpness(2,0.3), T.RandomAutocontrast(0.3),
        T.ConvertImageDtype(torch.float)
    ])

    dataloader = ds.pytorch(
        tensors=["images"],
        transform={ "images" : tform },
        num_workers=16, batch_size=config["batch_size"],
        decode_method={ "images":"numpy" },
        drop_last=True,
        buffer_size=4096,
        use_local_cache=True,
        shuffle=True
    )

    if config["compile"]:
        G, P = torch.compile(G), torch.compile(P)
    Gtx = torch.optim.AdamW(G.parameters(), lr = config["lr"], betas=(0.9, 0.99), weight_decay=1e-4)

    G, P, Gtx, dataloader = accelerator.prepare(G, P, Gtx, dataloader)

    for epoch in tqdm(range(42)):
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            losses = { "G" : 0 }
            reals = batch["images"]

            with accelerator.accumulate(G):
                output = GLoss(G, P, reals)
                losses["G"] = output["loss"]
                accelerator.backward(losses["G"])
                Gtx.step()
                Gtx.zero_grad()

            accelerator.log({ **output, **losses })

            if idx % config["log_every_n_steps"] == 0:
                with torch.no_grad():
                    fakes, _, _ = G(reals)
                    reals, fakes = reals.clamp(0,1), fakes.clamp(0,1)
                    accelerator.log({ "samples" : wandb.Image(fakes), "reals" : wandb.Image(reals) })


if __name__ == "__main__":
    main()