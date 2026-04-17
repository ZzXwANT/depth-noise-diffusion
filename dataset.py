"""
Dataset for depth noise diffusion training.

Loads real depth npy files, applies bilateral filtering to get pseudo-clean, computes residual as noise target. Includes augmentation for small datasets.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter


class DepthNoiseDataset(Dataset):
    """
    Yields (pseudo_clean, noise_residual) pairs, both float32 tensors of shape (1, H, W).

    Preprocessing:
      1. Load uint16 depth → float32, normalize to [0, 1] by dividing max uint16
      2. Gaussian blur (sigma=0.8) → pseudo_clean (保留更多噪声信息)
      3. residual = original - pseudo_clean
    """

    def __init__(self, data_dirs: list[str], augment: bool = True):
        self.paths = []
        for d in data_dirs:
            self.paths.extend(sorted(glob.glob(os.path.join(d, "*.npy"))))
        assert len(self.paths) > 0, f"No npy files found in {data_dirs}"
        self.augment = augment
        self.residual_scale = self._compute_residual_scale()
        print(f"✓ Residual scale (global std): {self.residual_scale:.6f}")

    def _compute_residual_scale(self, max_files: int = 200) -> float:
        """用部分文件估计残差的全局 std，作为归一化系数。"""
        paths = self.paths[:max_files]
        stds = []
        for p in paths:
            raw = np.load(p).astype(np.float32) / 65535.0
            clean = scipy_gaussian_filter(raw, sigma=0.8, mode="reflect")
            residual = raw - clean
            stds.append(residual.std())
        return float(np.mean(stds))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        raw = np.load(self.paths[idx]).astype(np.float32)
        raw = raw / 65535.0  # uint16 → [0, 1]

        # 使用高斯模糊保留噪声细节（mode='reflect'更快）
        clean = scipy_gaussian_filter(raw, sigma=0.8, mode="reflect").astype(np.float32)
        residual = (raw - clean) / self.residual_scale  # ✅ 归一化

        # to tensors (1, H, W)
        clean_t = torch.from_numpy(clean).unsqueeze(0)
        resid_t = torch.from_numpy(residual).unsqueeze(0)

        if self.augment:
            # random horizontal flip
            if torch.rand(1).item() > 0.5:
                clean_t = clean_t.flip(-1)
                resid_t = resid_t.flip(-1)
            # random vertical flip
            if torch.rand(1).item() > 0.5:
                clean_t = clean_t.flip(-2)
                resid_t = resid_t.flip(-2)

        return clean_t, resid_t
