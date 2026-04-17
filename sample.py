"""
Inference & visualization for trained depth noise diffusion model.

Usage:
    python -m depth_noise_diffusion.sample \
        --checkpoint runs/depth_noise_v2/best.pt \
        --data_dirs depth_dataset/20260409_200043/depth_real \
        --n_samples 4 --out_dir runs/depth_noise_v2/vis
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from .model import UNet, DDPM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="runs/depth_noise_diffusion/vis")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="噪声幅度缩放因子(0.0-2.0)，1.0为原始幅度，<1.0生成更小噪声，>1.0生成更大噪声",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load model
    unet = UNet(in_ch=2, out_ch=1)
    ddpm = DDPM(unet).to(device)
    ddpm.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    ddpm.eval()

    # load a few real frames, build pseudo-clean, generate noise
    import glob, json

    paths = []
    for d in args.data_dirs:
        paths.extend(sorted(glob.glob(os.path.join(d, "*.npy"))))
    paths = paths[: args.n_samples]

    fig, axes = plt.subplots(args.n_samples, 4, figsize=(16, 4 * args.n_samples))
    if args.n_samples == 1:
        axes = axes[None, :]

    # 加载归一化
    with open(os.path.join(os.path.dirname(args.checkpoint), "config.json")) as f:
        cfg = json.load(f)
    residual_scale = cfg["residual_scale"]
    for i, path in enumerate(paths):
        raw = np.load(path).astype(np.float32) / 65535.0
        clean = gaussian_filter(raw, sigma=0.8).astype(np.float32)
        real_residual = raw - clean

        clean_t = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).to(device)
        gen_residual = (
            ddpm.sample(clean_t, temperature=args.temperature).cpu().numpy()[0, 0]
        )
        gen_residual = gen_residual * residual_scale  # ✅ 反归一化
        gen_noisy_image = clean + gen_residual

        axes[i, 0].imshow(raw, cmap="viridis")
        axes[i, 0].set_title("Real depth")
        axes[i, 1].imshow(clean, cmap="viridis")
        axes[i, 1].set_title("Pseudo-clean")
        axes[i, 2].imshow(real_residual, cmap="RdBu", vmin=-0.02, vmax=0.02)
        axes[i, 2].set_title("Real noise")
        axes[i, 3].imshow(gen_residual, cmap="RdBu", vmin=-0.02, vmax=0.02)
        axes[i, 3].set_title("Generated noise")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {out_path}")

    # print noise statistics comparison
    print("\n── Noise statistics ──")
    print(f"{'':>15s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
    print(
        f"{'Real noise':>15s} {real_residual.mean():>10.6f} {real_residual.std():>10.6f} "
        f"{real_residual.min():>10.6f} {real_residual.max():>10.6f}"
    )
    print(
        f"{'Gen noise':>15s} {gen_residual.mean():>10.6f} {gen_residual.std():>10.6f} "
        f"{gen_residual.min():>10.6f} {gen_residual.max():>10.6f}"
    )
    print(f"\n✓ 生成采样参数: temperature={args.temperature}")


if __name__ == "__main__":
    main()
