# Depth Noise Diffusion

[English](./readme.md) | [简体中文](./readme.zh-CN.md)

| 实际深度图推理 | 仿真深度图推理（作为 `clean_depth`） |
| :--: | :--: |
| <img src="./assets/real_depth_inference_preview.png" width="420" alt="实际深度图推理结果"> | <img src="./assets/sim_clean_depth_inference_preview.png" width="420" alt="仿真深度图推理结果"> |

- 实际深度图推理：使用真实深度 `.npy`，先通过高斯滤波得到 `pseudo_clean`，再生成噪声残差并与真实残差对比。
- 仿真深度图推理：从 `sim_seed0.mp4` 抽取 4 帧，转换为 `128x128 uint16` 深度 `.npy`，直接作为 `clean_depth` 条件输入生成噪声。

本项目实现了用于深度相机噪声生成的条件扩散模型（DDPM），提供了完整的 sim2real 深度噪声训练与采样流程。

## 特性

- 轻量级条件 UNet + DDPM
- 基于残差噪声建模（`raw - pseudo_clean`）
- 支持梯度累积，便于使用更大有效 batch
- 支持混合精度训练（`--mixed_precision`）
- 自动保存 checkpoint 与训练配置（`config.json`）
- 提供采样可视化及噪声统计输出

## 文件说明

- `model.py` - 轻量级条件 UNet + DDPM 实现
- `dataset.py` - 深度噪声数据预处理与增强
- `train_v2.py` - 主训练脚本（推荐）
- `train_v1.py` - 旧版训练脚本（兼容保留）
- `sample.py` - 推理与可视化脚本
- `assets/` - README 展示图与推理结果示例
- `requirements.txt` - Python 依赖

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练（主脚本：`train_v2.py`）

```bash
python -m depth_noise_diffusion.train_v2 \
    --data_dirs path/to/depth_data1 path/to/depth_data2 \
    --epochs 800 \
    --batch_size 64 \
    --lr 2e-4 \
    --save_dir runs/depth_noise_v2
```

可选（显存/性能优化）：

```bash
python -m depth_noise_diffusion.train_v2 \
    --data_dirs path/to/depth_data \
    --batch_size 32 \
    --grad_accumulate_steps 2 \
    --mixed_precision
```

### 推理与可视化

```bash
python -m depth_noise_diffusion.sample \
    --checkpoint runs/depth_noise_v2/best.pt \
    --data_dirs path/to/test_data \
    --n_samples 4 \
    --out_dir runs/depth_noise_v2/vis
```

## 数据格式

- 输入：`uint16` 的 `.npy` 深度图
- 归一化：除以 `65535.0` 映射到 `[0, 1]`
- 伪干净图：高斯滤波（`sigma=0.8`）
- 噪声目标：原始深度与伪干净深度的残差

## 备注

- `train_v2.py` 为当前默认且持续维护的训练脚本。
- `train_v1.py` 为历史版本，保留用于参考与兼容。

## 许可证

本项目采用 [MIT License](../LICENSE) 开源协议。
