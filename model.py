"""
Lightweight conditional DDPM for depth-camera noise generation.

Architecture:
  - Small UNet (128→16 spatial, channels [64,128,256,512])
  - Condition: clean depth concatenated at input (2ch)
  - Time embedding: sinusoidal + MLP
  - Predicts epsilon (noise) with MSE loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time embedding ──────────────────────────────────────────────────────────


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeMLPEmb(nn.Module):
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalTimeEmb(time_dim),
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)


# ── UNet blocks ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ── UNet ────────────────────────────────────────────────────────────────────


class UNet(nn.Module):
    """Small UNet for 128×128 single-channel depth noise prediction."""

    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 1,
        ch_mult=(1, 2, 4, 8),
        base_ch: int = 64,
        time_dim: int = 256,
    ):
        super().__init__()
        self.time_emb = TimeMLPEmb(time_dim, time_dim)
        channels = [base_ch * m for m in ch_mult]  # [64, 128, 256, 512]

        # encoder
        self.in_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(ResBlock(channels[i], channels[i + 1], time_dim))
            self.down_samples.append(Downsample(channels[i + 1]))

        # bottleneck
        self.mid = ResBlock(channels[-1], channels[-1], time_dim)

        # decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_samples.append(Upsample(channels[i]))
            self.up_blocks.append(
                ResBlock(channels[i] + channels[i - 1], channels[i - 1], time_dim)
            )

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]), nn.SiLU(), nn.Conv2d(channels[0], out_ch, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = self.in_conv(x)
        skips = []
        for blk, down in zip(self.down_blocks, self.down_samples):
            skips.append(h)  # save BEFORE channel expansion
            h = blk(h, t_emb)
            h = down(h)
        h = self.mid(h, t_emb)
        for up, blk in zip(self.up_samples, self.up_blocks):
            h = up(h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = blk(h, t_emb)
        return self.out_conv(h)


# ── DDPM schedule & sampling ────────────────────────────────────────────────


class DDPM(nn.Module):
    def __init__(
        self,
        unet: UNet,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.unet = unet
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())

        # 后验方差（用于采样）
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]]).to(betas.device)
        self.register_buffer(
            "posterior_var", (betas**2) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        )
        self.register_buffer(
            "posterior_log_var_clipped",
            torch.log(torch.clamp(self.posterior_var, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1", betas * alpha_bar_prev.sqrt() / (1.0 - alpha_bar)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_bar_prev) * alphas.sqrt() / (1.0 - alpha_bar),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bar[t][:, None, None, None]
        b = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return a * x0 + b * noise, noise

    # def loss(self, noise_residual: torch.Tensor, clean_depth: torch.Tensor):
    #     """Training loss: predict epsilon from noised residual conditioned on clean depth.

    #     专注于主loss，让模型学到正确的噪声分布。
    #     """
    #     B = noise_residual.shape[0]
    #     t = torch.randint(0, self.T, (B,), device=noise_residual.device)
    #     noised, eps = self.q_sample(noise_residual, t)
    #     x_in = torch.cat([noised, clean_depth], dim=1)  # (B, 2, H, W)
    #     eps_pred = self.unet(x_in, t)

    #     # MSE loss: 直接预测epsilon
    #     mse_loss = F.mse_loss(eps_pred, eps)

    #     return mse_loss
    def loss(self, noise_residual: torch.Tensor, clean_depth: torch.Tensor):
        B = noise_residual.shape[0]
        t = torch.randint(0, self.T, (B,), device=noise_residual.device)
        noised, eps = self.q_sample(noise_residual, t)
        x_in = torch.cat([noised, clean_depth], dim=1)
        eps_pred = self.unet(x_in, t)

        # ✅ 按 epsilon 幅值加权，大噪声区域权重更高
        with torch.no_grad():
            weights = (eps.abs() / (eps.abs().mean() + 1e-8)).clamp(0.5, 3.0)

        loss = (weights * (eps_pred - eps) ** 2).mean()
        return loss

    @torch.no_grad()
    def sample(
        self, clean_depth: torch.Tensor, steps: int = None, temperature: float = 1.0
    ):
        """Reverse diffusion: generate noise residual conditioned on clean depth.

        Args:
            clean_depth: 条件信息（干净的深度图）
            steps: 采样步数（默认使用全部T步）
            temperature: 噪声缩放因子 (>0)，1.0为标准幅度，>1.0生成更大噪声，<1.0生成更小噪声
                        在采样初始化和每一步噪声中应用，确保充分利用模型容量
        """
        T = steps or self.T
        B, _, H, W = clean_depth.shape

        # 从温度调整的高斯开始（确保初始噪声有足够幅度）
        x = torch.randn(B, 1, H, W, device=clean_depth.device) * temperature

        for i in reversed(range(T)):
            t = torch.full((B,), i, device=x.device, dtype=torch.long)
            x_in = torch.cat([x, clean_depth], dim=1)
            eps_pred = self.unet(x_in, t)

            alpha = self.alphas[i]
            alpha_bar = self.alpha_bar[i]
            x = (1 / alpha.sqrt()) * (
                x - (1 - alpha) / (1 - alpha_bar).sqrt() * eps_pred
            )

            if i > 0:
                # 采样过程中也应用temperature，保持整个过程的噪声幅度
                noise_var = self.posterior_var[i]
                noise_scale = temperature * noise_var.sqrt()
                x = x + noise_scale * torch.randn_like(x)

        return x
