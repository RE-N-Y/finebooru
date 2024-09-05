import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchmetrics import Metric
from torchmetrics.image import FrechetInceptionDistance, StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, InceptionScore


import click
import deeplake
from tqdm import tqdm
from einops import rearrange
from accelerate import Accelerator
from modeling.base import CVQVAE

# converts image tensor of range -1,1 to 0~255 uint8
def toimage(image):
    image = image * 0.5 + 0.5
    image = image.clamp(0,1) * 255
    image = image.to(torch.uint8)
    return image

class CodebookUsage(Metric):
    def __init__(self, pages:int, **kwargs):
        super().__init__(**kwargs)
        self.pages = pages
        self.add_state("usage", default=torch.zeros(pages, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, idxes):
        self.usage += torch.bincount(idxes.view(-1), minlength=self.pages)
        self.total += len(idxes)

    def compute(self):
        return torch.sum(self.usage > 0) / self.pages

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
        T.ToTensor(), T.Lambda(lambda x: x.repeat(int( 3 / len(x)), 1, 1)),
        T.Resize((config["size"],config["size"]), antialias=True),
        T.ConvertImageDtype(torch.float), T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    dataloader = ds.pytorch(
        tensors=["images"],
        transform={ "images" : tform },
        num_workers=16, batch_size=config["batch_size"],
        decode_method={ "images":"numpy" }
    )


    PSNR = PeakSignalNoiseRatio()
    FID = FrechetInceptionDistance()
    IS = InceptionScore()
    USAGE = CodebookUsage(pages=G.quantiser.pages)

    G, PSNR, FID, IS, USAGE, dataloader = accelerator.prepare(G, PSNR, FID, IS, USAGE, dataloader)

    for batch in tqdm(dataloader, total=len(dataloader)):
        images = batch["images"]
        with torch.inference_mode():
            fakes, compress, idxes= G(images)
            images, fakes = toimage(images), toimage(fakes)

            FID.update(fakes, real=False)
            FID.update(images, real=True)
            PSNR.update(fakes, images)
            IS.update(fakes)
            USAGE.update(idxes)


    iscore, _ = IS.compute()
    accelerator.log({
        "FID": FID.compute(),
        "PSNR": PSNR.compute(),
        "IS": iscore,
        "usage": USAGE.compute()
    })


if __name__ == "__main__":
    main()
