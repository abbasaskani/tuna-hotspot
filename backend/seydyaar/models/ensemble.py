from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

def ensemble_stats(models: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    models: list of probability arrays (H,W) in 0..1

    Returns:
      agreement (0..1): fraction of models above threshold (default 0.6) OR based on closeness
      spread (0..1): std across models (already 0..~0.3 typically)
    """
    stack = np.stack(models, axis=0).astype(np.float32)
    spread = np.nanstd(stack, axis=0).astype(np.float32)

    thr = 0.6
    agree = np.mean((stack >= thr).astype(np.float32), axis=0).astype(np.float32)
    return agree, spread

def weighted_ensemble(models: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    # normalize
    wsum = sum(max(float(v), 0.0) for v in weights.values())
    if wsum <= 0:
        wsum = 1.0
        weights = {k: 1.0 for k in models.keys()}
    out = None
    for name, arr in models.items():
        w = max(float(weights.get(name, 0.0)), 0.0) / wsum
        out = arr * w if out is None else out + arr * w
    return np.clip(out, 0.0, 1.0).astype(np.float32)
