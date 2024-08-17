# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import GbufferDataset

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=lambda storage, loc: storage)['model'])
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])
    dataset = GbufferDataset(noisy_img_path=args.noisy_img_path, ground_truth_path=args.ground_truth_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    for noisy_img, Galbedo, Gdepth, Gnormal, ground_truth in loader:
        z = torch.randn(args.batch_size, 4, latent_size, latent_size, device=device)
        noisy_img, Galbedo, Gdepth, Gnormal, ground_truth = noisy_img.to(device), Galbedo.to(device), Gdepth.to(device), Gnormal.to(device), ground_truth.to(device)
        save_image(noisy_img, 'noisy_img.png', nrow=4, normalize=True, value_range=(-1, 1))
        save_image(ground_truth, 'ground_truth.png', nrow=4, normalize=True, value_range=(-1, 1))
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            noisy_img = vae.encode(noisy_img).latent_dist.sample().mul_(0.18215)
            Galbedo = vae.encode(Galbedo).latent_dist.sample().mul_(0.18215)
            Gdepth = vae.encode(Gdepth).latent_dist.sample().mul_(0.18215)
            Gnormal = vae.encode(Gnormal).latent_dist.sample().mul_(0.18215)
            ground_truth = vae.encode(ground_truth).latent_dist.sample().mul_(0.18215)
        model_kwargs = dict(noisy_img=noisy_img, Galbedo=Galbedo, Gdepth=Gdepth, Gnormal=Gnormal)
        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        # Save and display images:
        save_image(samples, "sample_steps{}.png".format(args.num_sampling_steps), nrow=4, normalize=True, value_range=(-1, 1))
        break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)