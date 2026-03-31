from __future__ import annotations

from pathlib import Path
from typing import Any


def safe_token(raw: Any) -> str:
    """Sanitize a value for safe use in filenames.

    Keeps alphanumerics and a small set of punctuation that is generally safe across
    filesystems. All other characters are replaced with underscores.
    """
    s = str(raw)
    return "".join(ch if ch.isalnum() or ch in "-_.=" else "_" for ch in s)


def dump_stim_circuit(circuit: Any, *, out_dir: Path, name: str) -> Path:
    """Write a stim circuit to disk using a sanitized filename.

    Returns the path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe_token(name)}.stim"
    path.write_text(str(circuit), encoding="utf-8")
    return path
