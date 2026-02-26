from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from bm3d import bm3d as _bm3d
    from bm3d import BM3DStages as _BM3DStages

    _HAS_BM3D = True
except Exception:
    _HAS_BM3D = False
    _bm3d = None
    _BM3DStages = None

try:
    from skimage.restoration import denoise_wavelet

    _HAS_WAVELET = True
except Exception:
    _HAS_WAVELET = False
    denoise_wavelet = None


class BM3D(nn.Module):
    def __init__(self, sigma_psd: float = 0.05):
        super().__init__()
        self.sigma_psd = float(sigma_psd)

    def _denoise_numpy(self, frame: np.ndarray) -> np.ndarray:
        frame = np.clip(frame.astype(np.float32), 0.0, 1.0)
        if _HAS_BM3D:
            out = _bm3d(
                frame, sigma_psd=self.sigma_psd, stage_arg=_BM3DStages.ALL_STAGES
            )
            return np.clip(out.astype(np.float32), 0.0, 1.0)
        if _HAS_WAVELET:
            try:
                out = denoise_wavelet(
                    frame, channel_axis=None, rescale_sigma=True, mode="soft"
                )
                return np.clip(out.astype(np.float32), 0.0, 1.0)
            except Exception:
                pass

        import cv2

        out = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
        return np.clip(out.astype(np.float32), 0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x[:, x.size(1) // 2]
        if x.size(1) != 1:
            return x

        arr = x.detach().cpu().numpy().astype(np.float32)
        outs = [self._denoise_numpy(arr[i, 0]) for i in range(arr.shape[0])]
        out = (
            torch.from_numpy(np.stack(outs, axis=0))
            .to(device=x.device, dtype=x.dtype)
            .unsqueeze(1)
        )
        return torch.clamp(out, 0.0, 1.0)
