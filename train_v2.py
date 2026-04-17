"""
改进版训练脚本 - 支持大batch、梯度累积、混合精度

Usage:
    # 基础运行
    python -m depth_noise_diffusion.train_v2 \
        --data_dirs depth_dataset/20260409_200043/depth_real depth_dataset/20260409_203800/depth_real/depth_dataset/20260412_152525/depth_real \
        --epochs 800 --batch_size 64 --lr 2e-4

    # 使用梯度累积 (当显存不足时)
    python -m depth_noise_diffusion.train_v2 \
        --batch_size 32 --grad_accumulate_steps 2  # 有效batch = 64

    # 启用混合精度 (加速+省显存)
    python -m depth_noise_diffusion.train_v2 \
        --batch_size 64 --mixed_precision
"""

import json
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

from .model import UNet, DDPM
from .dataset import DepthNoiseDataset


def parse_args():
    p = argparse.ArgumentParser(description="改进的深度噪声DDPM训练脚本")

    # 必要参数
    p.add_argument("--data_dirs", nargs="+", required=True, help="数据目录列表")

    # 训练超参数
    p.add_argument("--epochs", type=int, default=800, help="训练epochs数")
    p.add_argument("--batch_size", type=int, default=64, help="per-GPU batch size")
    p.add_argument("--lr", type=float, default=2e-4, help="学习率")
    p.add_argument(
        "--weight_decay", type=float, default=1e-4, help="AdamW weight decay"
    )

    # 显存优化
    p.add_argument(
        "--grad_accumulate_steps",
        type=int,
        default=1,
        help="梯度累积步数 (有效batch=batch_size*grad_accumulate_steps)",
    )
    p.add_argument(
        "--mixed_precision", action="store_true", help="使用混合精度训练 (fp16)"
    )

    # DDPM参数
    p.add_argument("--T", type=int, default=1000, help="扩散过程的时间步数")

    # 保存和日志
    p.add_argument(
        "--save_dir", type=str, default="runs/depth_noise_v2", help="checkpoint保存目录"
    )
    p.add_argument("--save_every", type=int, default=1000, help="每N个epoch保存一次")
    p.add_argument("--log_every", type=int, default=10, help="每N个epoch打印日志")

    # 其他
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader worker数")

    return p.parse_args()


def log_config(args, residual_scale, save_dir):
    """保存训练配置"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.grad_accumulate_steps,
        "lr": args.lr,
        "epochs": args.epochs,
        "T": args.T,
        "mixed_precision": args.mixed_precision,
        "grad_accumulate_steps": args.grad_accumulate_steps,
        "num_data_dirs": len(args.data_dirs),
        "residual_scale": residual_scale,
    }

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 配置日志
    print("=" * 70)
    print("深度噪声DDPM - 改进训练脚本")
    print("=" * 70)
    print(f"\n【基础配置】")
    print(f"  数据目录: {len(args.data_dirs)} 个")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  有效Batch Size: {args.batch_size * args.grad_accumulate_steps}")
    print(f"  学习率: {args.lr:.2e}")
    print(f"  DDPM时间步: {args.T}")

    if args.mixed_precision:
        print(f"  混合精度: 启用 (fp16)")
    else:
        print(f"  混合精度: 禁用")

    if args.grad_accumulate_steps > 1:
        print(f"  梯度累积: {args.grad_accumulate_steps} steps")

    print(f"  设备: {device}")
    print()

    # ==================== 数据加载 ====================
    print("[1/4] 加载数据...")
    dataset = DepthNoiseDataset(args.data_dirs, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # 丢弃最后不完整的batch
    )

    total_samples = len(dataset)
    total_batches = len(loader)
    print(f"✓ 样本数: {total_samples}")
    print(f"✓ Batches/Epoch: {total_batches}")
    print()

    # ==================== 模型 ====================
    print("[2/4] 初始化模型...")
    unet = UNet(in_ch=2, out_ch=1)
    ddpm = DDPM(unet, T=args.T).to(device)

    # 计算参数量
    param_count = sum(p.numel() for p in ddpm.parameters())
    print(f"✓ 模型参数: {param_count/1e6:.1f}M")

    # ==================== 优化器 ====================
    print("[3/4] 设置优化器...")
    opt = torch.optim.AdamW(
        ddpm.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # 混合精度
    if args.mixed_precision:
        scaler = torch.amp.GradScaler("cuda")
        print(f"✓ 混合精度: fp16 enabled")
    else:
        scaler = None

    print()

    # ==================== 训练循环 ====================
    print("[4/4] 开始训练...")
    print("=" * 70)

    # 新增归一化噪声sclae
    log_config(args, dataset.residual_scale, args.save_dir)

    # 用于记录
    best_loss = float("inf")
    loss_history = []

    try:
        for epoch in range(1, args.epochs + 1):
            ddpm.train()
            total_loss = 0.0
            opt.zero_grad()

            pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

            for batch_idx, (clean, residual) in enumerate(pbar):
                clean = clean.to(device)
                residual = residual.to(device)

                # Forward pass
                if args.mixed_precision:
                    with torch.amp.autocast("cuda"):
                        loss = ddpm.loss(residual, clean)

                    # Backward pass
                    scaler.scale(loss).backward()

                    # 梯度累积
                    if (batch_idx + 1) % args.grad_accumulate_steps == 0:
                        # 梯度裁剪
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)

                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad()
                else:
                    loss = ddpm.loss(residual, clean)
                    loss.backward()

                    # 梯度累积
                    if (batch_idx + 1) % args.grad_accumulate_steps == 0:
                        torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
                        opt.step()
                        opt.zero_grad()

                total_loss += loss.item()
                pbar.update()

            # 学习率调整
            scheduler.step()

            avg_loss = total_loss / len(loader)
            loss_history.append(avg_loss)

            # 日志
            if epoch % args.log_every == 0 or epoch == 1:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.6f} | lr={lr:.2e}"
                )

                # GPU显存监控
                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / (1024**3)
                    mem_max = torch.cuda.max_memory_allocated() / (1024**3)
                    print(
                        f"             | GPU mem: {mem_used:.1f}GB / max {mem_max:.1f}GB"
                    )

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ddpm.state_dict(), os.path.join(args.save_dir, "best.pt"))
                if args.log_every <= 10:
                    print(f"             ⭐ 新纪录: {best_loss:.6f}")

            # 定期保存
            if epoch % args.save_every == 0:
                torch.save(
                    ddpm.state_dict(), os.path.join(args.save_dir, f"epoch_{epoch}.pt")
                )

        # 训练完成
        # torch.save(ddpm.state_dict(), os.path.join(args.save_dir, "final.pt"))

        # 保存loss历史
        with open(os.path.join(args.save_dir, "loss_history.json"), "w") as f:
            json.dump(loss_history, f)

        print("=" * 70)
        print("✓ 训练完成!")
        print(f"  最佳Loss: {best_loss:.6f}")
        print(f"  模型保存到: {args.save_dir}")
        print()

        print("【推理命令】")
        print(f"python -m depth_noise_diffusion.sample \\")
        print(f"  --checkpoint {args.save_dir}/best.pt \\")
        print(f"  --data_dirs depth_dataset/20260409_200043/depth_real \\")
        print(f"  --temperature 0.5 \\")
        print(f"  --out_dir {args.save_dir}/vis")
        print()

    except KeyboardInterrupt:
        print("\n⚠ 训练被中断")
        print(f"  已保存最佳模型: {os.path.join(args.save_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
