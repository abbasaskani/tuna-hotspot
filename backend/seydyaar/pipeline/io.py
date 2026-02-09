from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np

def write_bin_f32(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype=np.float32).tofile(path)

def write_bin_u8(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype=np.uint8).tofile(path)

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def minify_json_for_web(path: Path) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
