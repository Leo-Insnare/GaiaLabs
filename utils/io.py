from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def make_output_dir(base_dir: str | Path, run_name: str | None = None) -> Path:
    base = ensure_dir(base_dir)
    if run_name:
        return ensure_dir(base / safe_name(run_name))
    idx = 1
    while True:
        candidate = base / f"run_{idx:03d}"
        if not candidate.exists():
            return ensure_dir(candidate)
        idx += 1


def safe_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "run"


def zip_outputs(output_dir: str | Path, zip_path: str | Path | None = None) -> Path:
    output_dir = Path(output_dir)
    if zip_path is None:
        zip_path = output_dir.with_suffix(".zip")
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path.is_file() and path.resolve() != zip_path.resolve():
                zf.write(path, path.relative_to(output_dir))
    return zip_path


def read_bytes(path: str | Path) -> bytes:
    with Path(path).open("rb") as f:
        return f.read()
