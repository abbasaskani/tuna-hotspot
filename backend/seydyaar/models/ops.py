from __future__ import annotations

from typing import Dict
import numpy as np

from .scoring import score_current_m_s, score_waves_hs


def ops_feasibility(
    current_m_s: np.ndarray,
    waves_hs_m: np.ndarray,
    priors: Dict,
    gear_depth_m: float = 10.0,
) -> np.ndarray:
    """Operational feasibility Pops (0..1).

    Notes:
    - This is a *soft* ops layer (continuous) rather than hard masking.
    - `gear_depth_m` (5/10/15/20) is used as a simple, defensible knob:
        - shallower gear → waves matter more
        - deeper gear → currents matter slightly more

    For real production, calibrate this using gear-specific logs + feedback.
    """

    # Waves soft max
    soft_max = float(priors.get("waves_hs_soft_max_m", 1.5))
    s_w = score_waves_hs(waves_hs_m, soft_max_m=soft_max)

    # Currents: prefer moderate currents (or penalize too strong)
    opt = float(priors.get("current_opt_m_s", 0.4))
    sig = float(priors.get("current_sigma_m_s", 0.25))
    s_c = score_current_m_s(current_m_s, opt_m_s=opt, sigma_m_s=sig)

    # Depth-aware weights (light-touch): normalize to sum=1
    d = float(gear_depth_m)
    # waves weight in [0.45, 0.65] roughly
    w_waves = 0.55 + (10.0 - d) * 0.01
    w_curr = 0.45 + (d - 10.0) * 0.01
    w_waves = float(np.clip(w_waves, 0.40, 0.70))
    w_curr = float(np.clip(w_curr, 0.30, 0.60))
    s = w_waves + w_curr
    w_waves /= s
    w_curr /= s

    pops = np.clip(w_waves * s_w + w_curr * s_c, 0.0, 1.0)
    return pops
