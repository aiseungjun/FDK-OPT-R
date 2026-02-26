from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


class NLM(nn.Module):
    def __init__(
        self,
        h: float = 7.0,
        template_window_size: int = 7,
        search_window_size: int = 21,
    ):
        super().__init__()
        self.h = float(h)
        self.template_window_size = int(template_window_size)
        self.search_window_size = int(search_window_size)

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        x = np.clip(frame, 0.0, 1.0)
        u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        out = cv2.fastNlMeansDenoising(
            u8,
            None,
            h=self.h,
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size,
        )
        return out.astype(np.float32) / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x[:, x.size(1) // 2]
        if x.size(1) != 1:
            return x

        arr = x.detach().cpu().numpy().astype(np.float32)
        outs = [self._denoise(arr[i, 0]) for i in range(arr.shape[0])]
        out = (
            torch.from_numpy(np.stack(outs, axis=0))
            .to(device=x.device, dtype=x.dtype)
            .unsqueeze(1)
        )
        return torch.clamp(out, 0.0, 1.0)
