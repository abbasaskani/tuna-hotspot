from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def score_temp_c(sst_c: np.ndarray, opt_c: float, sigma_c: float) -> np.ndarray:
    return _gauss(sst_c, opt_c, sigma_c)

def score_chl_mg_m3(chl: np.ndarray, opt_mg_m3: float, sigma_log10: float) -> np.ndarray:
    chl = np.clip(chl, 1e-6, None)
    return _gauss(np.log10(chl), np.log10(opt_mg_m3), sigma_log10)

def score_current_m_s(spd: np.ndarray, opt_m_s: float, sigma_m_s: float) -> np.ndarray:
    return _gauss(spd, opt_m_s, sigma_m_s)

def score_waves_hs(hs_m: np.ndarray, soft_max_m: float = 1.5, softness: float = 0.35) -> np.ndarray:
    # logistic penalty: ~1 below soft_max, declines above
    return 1.0 / (1.0 + np.exp((hs_m - soft_max_m) / max(softness, 1e-6)))

def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(arr.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)

def front_score(temp_front: np.ndarray, chl_front: np.ndarray, ssh_front: np.ndarray,
                w_temp: float = 0.5, w_chl: float = 0.25, w_ssh: float = 0.25) -> np.ndarray:
    s = w_temp * temp_front + w_chl * chl_front + w_ssh * ssh_front
    # normalize 0..1 robustly
    lo, hi = np.nanpercentile(s, 5), np.nanpercentile(s, 95)
    out = (s - lo) / (hi - lo + 1e-9)
    return np.clip(out, 0.0, 1.0)

@dataclass
class HabitatInputs:
    sst_c: np.ndarray
    chl_mg_m3: np.ndarray
    current_m_s: np.ndarray
    waves_hs_m: np.ndarray
    ssh_m: np.ndarray  # optional (demo can use zeros)

def habitat_scoring(inputs: HabitatInputs, priors: Dict, weights: Dict) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      phabitat (0..1), components dict for explainability.
    """
    s_temp = score_temp_c(inputs.sst_c, priors["sst_opt_c"], priors["sst_sigma_c"])
    s_chl  = score_chl_mg_m3(inputs.chl_mg_m3, priors["chl_opt_mg_m3"], priors["chl_sigma_log10"])
    s_cur  = score_current_m_s(inputs.current_m_s, priors["current_opt_m_s"], priors["current_sigma_m_s"])

    # fronts from gradients (temp/chl/ssh)
    tf = gradient_magnitude(inputs.sst_c)
    cf = gradient_magnitude(np.log10(np.clip(inputs.chl_mg_m3, 1e-6, None)))
    sf = gradient_magnitude(inputs.ssh_m)
    fw = priors.get("front_weights", {"temp":0.5,"chl":0.25,"ssh":0.25})
    s_front = front_score(tf, cf, sf, fw.get("temp",0.5), fw.get("chl",0.25), fw.get("ssh",0.25))

    # operational score (waves) kept separate in Pops, but we still compute component here for diagnostics
    s_waves = score_waves_hs(inputs.waves_hs_m, priors.get("waves_hs_soft_max_m", 1.5))

    # weights (auto-renormalize)
    w = dict(weights)
    total = sum(max(v, 0.0) for v in w.values())
    if total <= 0:
        w = {"temp":1.0}
        total = 1.0
    for k in list(w.keys()):
        w[k] = max(float(w[k]), 0.0) / total

    phab = (
        w.get("temp",0.0)*s_temp +
        w.get("chl",0.0)*s_chl +
        w.get("front",0.0)*s_front +
        w.get("current",0.0)*s_cur
    )
    phab = np.clip(phab, 0.0, 1.0)

    comps = {
        "score_temp": s_temp,
        "score_chl": s_chl,
        "score_front": s_front,
        "score_current": s_cur,
        "score_waves": s_waves,
        "temp_front": tf,
        "chl_front": cf,
        "ssh_front": sf
    }
    return phab, comps
