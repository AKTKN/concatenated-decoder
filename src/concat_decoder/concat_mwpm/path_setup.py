from __future__ import annotations

import sys
from pathlib import Path


def ensure_color_code_stim_import(project_root: Path | None = None) -> Path:
    """Add local color-code-stim source path to sys.path when available."""
    if project_root is None:
        # src-layout: <repo>/src/concat_decoder/concat_mwpm/...
        project_root = Path(__file__).resolve().parents[3]

    src_path = project_root / "color-code-stim" / "src"
    src_path_str = str(src_path)
    if src_path.exists() and src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)
    return project_root
