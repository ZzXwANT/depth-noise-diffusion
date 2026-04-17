"""
Training script for depth noise conditional DDPM.

Usage:
    python -m depth_noise_diffusion.train \
        --data_dirs depth_dataset/20260409_200043/depth_real depth_dataset/20260409_203800/depth_real \
        --epochs 800 --batch_size 8 --lr 2e-4 --save_dir runs/depth_noise_diffusion
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import UNet, DDPM
from .dataset import DepthNoiseDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=1000, help="diffusion timesteps")
    p.add_argument("--save_dir", type=str, default="runs/depth_noise_diffusion")
    p.add_argument(
        "--save_every", type=int, default=100, help="save checkpoint every N epochs"
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # data
    dataset = DepthNoiseDataset(args.data_dirs, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    # model
    unet = UNet(in_ch=2, out_ch=1)
    ddpm = DDPM(unet, T=args.T).to(device)
    opt = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # training loop
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        ddpm.train()
        total_loss = 0.0
        for clean, residual in loader:
            clean, residual = clean.to(device), residual.to(device)
            loss = ddpm.loss(residual, clean)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ddpm.state_dict(), os.path.join(args.save_dir, "best.pt"))

        if epoch % args.save_every == 0:
            torch.save(
                ddpm.state_dict(), os.path.join(args.save_dir, f"epoch_{epoch}.pt")
            )

    torch.save(ddpm.state_dict(), os.path.join(args.save_dir, "final.pt"))
    print(f"Training done. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
