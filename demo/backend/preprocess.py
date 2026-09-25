"""推論前處理：與實驗時 Anomalib 的前處理一致，但不依賴 PyTorch。

實驗時的前處理是 torchvision Resize(256, antialias=True)；Anomalib 匯出 ONNX 時內建的 Resize
卻是 antialias=False，直接用會讓分數偏移。因此這裡先以 PIL（逐通道浮點、含 antialias）縮到 256×256，
ONNX 內建的 Resize 就變成 256→256 的恆等運算，正規化與後處理仍由 ONNX 模型完成。
"""
from pathlib import Path

import numpy as np
from PIL import Image

INPUT_SIZE = 256


def load_input(path: Path | str) -> np.ndarray:
    """讀取影像並轉成 ONNX 輸入：float32、形狀 (1, 3, 256, 256)、數值 0～1。"""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    channels = [np.asarray(Image.fromarray(rgb[..., c], mode="F").resize((INPUT_SIZE, INPUT_SIZE),
                                                                           Image.BILINEAR))
                for c in range(3)]
    return np.stack(channels)[None].astype(np.float32)
