# Seyd‑Yaar (صیدیار) — Tuna Catch Probability PWA 🐟🌊

A lightweight **installable PWA** + a **Python generator** that produces:
- **Habitat Suitability** (Phabitat)
- **Operational Feasibility** (Pops)
- **Catch Probability** (Pcatch = Phabitat × Pops) ✅ default
- **Uncertainty** maps: **Agreement** + **Spread/Std**
- **Explainability**: per‑species profile (Skipjack/Yellowfin), weights, curves, top‑10 hotspots, covariate table
- **Audit / versioning** via `meta.json` (run time, data sources, model versions, QC/gap‑fill flags, missing, etc.)

> In this demo ZIP, the generator can create **synthetic demo data** (fast) so the UI works end‑to‑end offline.
> Production hooks are included for **AIS effort (Global Fishing Watch)**, **MaxEnt/PPP presence‑only**, and **GeoTIFF/COG export**.

---

## Quick start (Demo)

### 1) Python (generator)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Fast demo (few timesteps)
python -m seydyaar demo-generate --date today --fast
```

This writes outputs to `../docs/latest/...` (static website reads from there).

### 2) Open the PWA
Open `docs/index.html` (or serve `docs/`):
```bash
cd docs
python -m http.server 8080
```
Then open the printed local URL in your browser.

---

## Production notes (what’s scaffolded + needs real credentials/data)

- **AIS Effort proxy (Global Fishing Watch)**  
  Implemented as a provider + client wrapper, but needs **API key/token** and network.
- **MaxEnt / PPP real training**  
  Works when presence points exist (AIS proxy / CSV upload). Bias correction is supported (background sampling).
- **COG/GeoTIFF**  
  Demo writes GeoTIFF; COG‑style tiling/overviews included (no `rio-cogeo` dependency).

---

## Repo layout

- `docs/` — GitHub Pages static site (PWA)
- `docs/latest/` — generated data (meta + binaries)
- `backend/seydyaar/` — generator + models + providers

---

## Credits

Designed by: **عباس آسکانی — Abbas Askani**  
**Askani Fishing Data company**
