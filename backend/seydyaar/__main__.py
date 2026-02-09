"""Seyd‑Yaar CLI entrypoint.

- `demo-generate`: creates a lightweight (synthetic) demo run under docs/latest/
  so the PWA works fully offline.

For real runs, wire the pipeline to real data sources and schedule it.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _try_load_dotenv() -> None:
    """Load .env if python-dotenv is installed (optional)."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        return


def _parse_depths(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    _try_load_dotenv()

    parser = argparse.ArgumentParser(prog="seydyaar")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo-generate", help="Generate an offline demo run into docs/latest")
    p.add_argument("--date", default="today", help="Run date (YYYY-MM-DD) or 'today'")
    p.add_argument("--past-days", type=int, default=0, help="Past days to include (max 7 recommended)")
    p.add_argument("--future-days", type=int, default=0, help="Future days to include (max 4 recommended)")
    p.add_argument("--step-hours", type=int, default=6, help="Time step in hours (2 is 'default' for full mode)")
    p.add_argument("--fast", action="store_true", help="Fast demo (coarser grid, fewer background samples)")
    p.add_argument("--out", default=str(Path("docs") / "latest"), help="Output folder")

    # Presence proxy for MaxEnt/PPP
    p.add_argument(
        "--presence-mode",
        choices=["auto", "ais", "weak", "csv"],
        default="auto",
        help="Presence proxy mode for PPP/MaxEnt (auto tries AIS first if token available)",
    )
    p.add_argument("--presence-csv", default="", help="CSV path for presence-only points (optional)")

    # Optional GeoTIFF/COG export
    p.add_argument("--export-cog", action="store_true", help="Write per-time COG GeoTIFFs (larger output)")

    # Operational depth menu (precomputed)
    p.add_argument(
        "--depths",
        default="5,10,15,20",
        help="Comma-separated gear depths (m) to precompute for ops/catch maps",
    )

    args = parser.parse_args()

    if args.cmd == "demo-generate":
        from seydyaar.pipeline.demo_generate import demo_generate

        demo_generate(
            date=args.date,
            out_dir=args.out,
            past_days=args.past_days,
            future_days=args.future_days,
            step_hours=args.step_hours,
            fast=args.fast,
            presence_mode=args.presence_mode,
            presence_csv=(args.presence_csv or None),
            export_cog=bool(args.export_cog),
            depths_m=_parse_depths(args.depths),
        )


if __name__ == "__main__":
    main()
