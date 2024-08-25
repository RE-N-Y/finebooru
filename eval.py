import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.inception import InceptionScore


import click
import deeplake
from tqdm import tqdm
from accelerate import Accelerator
from modeling.base import CVQVAE

# converts image tensor of range -1,1 to 0~255 uint8
def toimage(image):
    image = image * 0.5 + 0.5
    image = image.clamp(0,1) * 255
    image = image.to(torch.uint8)
    return image

@click.command()
@click.option("--name", default="name", type=str)
@click.option("--dataset", default="hub://activeloop/imagenet-val", type=str)
@click.option("--size", default=256, type=int)
@click.option("--batch_size", default=64, type=int)
def main(**config):
    torch.set_float32_matmul_precision('high')
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers("vit", config, init_kwargs={"wandb":{"name":config["name"]}})

    ds = deeplake.load(config["dataset"])
    G = CVQVAE.from_pretrained(config["name"])
    G = torch.compile(G)
    G.eval()

    tform = T.Compose([
        T.ToTensor(), T.Resize(config["size"], antialias=True),
        T.RandomResizedCrop(config["size"], antialias=True),
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

    
    SSIM = StructuralSimilarityIndexMeasure()
    PSNR = PeakSignalNoiseRatio()
    FID = FrechetInceptionDistance()
    IS = InceptionScore()

    G, SSIM, PSNR, FID, IS, dataloader = accelerator.prepare(G, SSIM, PSNR, FID, IS, dataloader)

    for batch in tqdm(dataloader, total=len(dataloader)):
        images = batch["images"]
        with torch.inference_mode():
            fakes, compress, idxes= G(images)
            images, fakes = toimage(images), toimage(fakes)
            FID.update(fakes, real=False)
            FID.update(images, real=True)
            PSNR.update(fakes, images)
            IS.update(fakes)

    accelerator.log({
        "FID": FID.compute(),
        "PSNR": PSNR.compute(),
        "IS": IS.compute()
    })

    
if __name__ == "__main__":
    main()



