# Depth Noise Diffusion

This project implements a conditional Denoising Diffusion Probabilistic Model (DDPM) for generating depth camera noise. It provides a complete training pipeline for depth noise simulation in sim-to-real vision applications.

## Features

- **Lightweight UNet Architecture**: Efficient model with 128→8 spatial resolution and channel progression [64,128,256,512]
- **Conditional Generation**: Uses clean depth as conditioning input via channel concatenation
- **Noise Prediction**: Epsilon prediction with MSE loss, T=1000 timesteps
- **Data Processing**: Loads uint16 depth npy files, applies Gaussian filtering for pseudo-clean targets, computes residuals
- **Augmentation**: Includes flip augmentations for small datasets
- **Training Utilities**: Cosine LR scheduling, gradient clipping, automatic checkpoint saving
- **Visualization**: Inference script with 4-column comparison plots and statistics

## Files

- `model.py` — Lightweight conditional UNet + DDPM implementation
- `dataset.py` — Depth noise dataset with preprocessing and augmentation
- `train.py` — Training script with configurable hyperparameters
- `train_v2.py` — Alternative training script (check differences)
- `sample.py` — Inference and visualization script
- `requirements.txt` — Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python -m depth_noise_diffusion.train \
    --data_dirs path/to/depth_data1 path/to/depth_data2 \
    --epochs 800 \
    --batch_size 8 \
    --lr 2e-4 \
    --save_dir runs/depth_noise_diffusion
```

### Inference

```bash
python -m depth_noise_diffusion.sample \
    --checkpoint runs/depth_noise_diffusion/best.pt \
    --data_dirs path/to/test_data \
    --n_samples 4 \
    --out_dir runs/depth_noise_diffusion/vis
```

## Data Format

- Input: uint16 numpy arrays (.npy files) containing depth images
- Normalization: Divided by 65535.0 to [0,1] range
- Pseudo-clean: Generated via Gaussian filtering (sigma=0.8)
- Noise target: Residual between original and pseudo-clean

## License

[Add your license here]