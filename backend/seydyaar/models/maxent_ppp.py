from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

@dataclass
class PPPModel:
    coef: np.ndarray          # shape (F,)
    intercept: float
    feat_mean: np.ndarray     # (F,)
    feat_std: np.ndarray      # (F,)
    feature_names: Tuple[str, ...]

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd, mu, sd

def fit_presence_background_logit(
    X_presence: np.ndarray,
    X_bg: np.ndarray,
    l2: float = 1.0,
    steps: int = 250,
    lr: float = 0.15,
    seed: int = 42,
) -> PPPModel:
    """
    A lightweight MaxEnt/PPP‑style fit:
      Presence-only -> discriminate presence vs background with regularized logistic regression.

    Notes:
    - This is a pragmatic implementation for the product pipeline (no heavy deps).
    - For small datasets it works well; for production you may swap in a robust GLM/PPM library.
    """
    rng = np.random.default_rng(seed)

    X = np.vstack([X_presence, X_bg]).astype(np.float32)
    y = np.concatenate([np.ones(len(X_presence), dtype=np.float32), np.zeros(len(X_bg), dtype=np.float32)])

    Xs, mu, sd = _standardize(X)

    # init
    F = Xs.shape[1]
    w = rng.normal(0, 0.05, size=(F,)).astype(np.float32)
    b = 0.0

    # gradient descent (kept lightweight for demo/fast-runs)
    prev_loss: float | None = None
    stall = 0
    for _ in range(int(steps)):
        z = Xs @ w + b
        p = _sigmoid(z)
        # gradients
        err = (p - y)
        gw = (Xs.T @ err) / len(y) + l2 * w
        gb = float(np.mean(err))
        w -= lr * gw
        b -= lr * gb

        # simple early-stop: if loss stops improving, cut iterations
        # (prevents long BLAS loops on slower machines)
        eps = 1e-6
        loss = float(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean() + 0.5 * l2 * float(np.mean(w * w)))
        if prev_loss is not None:
            if abs(prev_loss - loss) < 1e-5:
                stall += 1
            else:
                stall = 0
        prev_loss = loss
        if stall >= 15:
            break

    return PPPModel(coef=w.astype(np.float32), intercept=float(b), feat_mean=mu.astype(np.float32), feat_std=sd.astype(np.float32), feature_names=tuple(f"f{i}" for i in range(F)))

def predict_prob(model: PPPModel, X: np.ndarray) -> np.ndarray:
    Xs = (X - model.feat_mean) / model.feat_std
    z = Xs @ model.coef + model.intercept
    return _sigmoid(z).astype(np.float32)

def build_feature_stack(
    sst_c: np.ndarray,
    chl_mg_m3: np.ndarray,
    current_m_s: np.ndarray,
    waves_hs_m: np.ndarray,
    front_score01: np.ndarray,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Returns X with shape (N, F), where N = H*W.
    """
    logchl = np.log10(np.clip(chl_mg_m3, 1e-6, None))
    feats = [
        sst_c,
        logchl,
        current_m_s,
        waves_hs_m,
        front_score01,
        sst_c ** 2,
        logchl ** 2,
    ]
    names = ("sst", "logchl", "cur", "waves", "front", "sst2", "logchl2")
    H, W = sst_c.shape
    X = np.stack([f.reshape(-1) for f in feats], axis=1).astype(np.float32)
    return X, names

def sample_points_from_mask(mask: np.ndarray, n: int, weights: Optional[np.ndarray] = None, seed: int = 0) -> np.ndarray:
    """
    Returns flat indices sampled from mask==1.
    If weights provided, it must be same shape and nonnegative (only used within mask).
    """
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask.reshape(-1) > 0)
    if len(idx) == 0:
        raise ValueError("AOI mask is empty")
    if weights is None:
        return rng.choice(idx, size=n, replace=(n > len(idx)))
    w = np.clip(weights.reshape(-1)[idx], 0.0, None).astype(np.float64)
    if np.all(w == 0):
        return rng.choice(idx, size=n, replace=(n > len(idx)))
    w = w / w.sum()
    return rng.choice(idx, size=n, replace=(n > len(idx)), p=w)

def fit_ppp_from_presence_proxy(
    X_grid: np.ndarray,
    mask: np.ndarray,
    presence_idx: np.ndarray,
    bias_surface: Optional[np.ndarray] = None,
    n_background: int = 4000,
    l2: float = 1.0,
    seed: int = 42,
) -> PPPModel:
    """
    Fit PPP using presence indices + background sampled (optionally) with bias correction.
    """
    N = X_grid.shape[0]
    # presence features
    pres = X_grid[presence_idx]
    # background indices sampled from AOI, optionally weighted by bias surface (effort)
    bg_idx = sample_points_from_mask(mask, n=n_background, weights=bias_surface, seed=seed+1)
    bg = X_grid[bg_idx]

    model = fit_presence_background_logit(pres, bg, l2=l2, seed=seed)
    model = PPPModel(coef=model.coef, intercept=model.intercept, feat_mean=model.feat_mean, feat_std=model.feat_std, feature_names=tuple(["sst","logchl","cur","waves","front","sst2","logchl2"]))
    return model
