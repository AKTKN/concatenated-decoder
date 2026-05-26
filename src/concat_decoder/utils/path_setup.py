from __future__ import annotations

import sys
from pathlib import Path


def ensure_local_imports(project_root: Path | None = None) -> Path:
    """Ensure local editable-like imports work without pip installation.

    Adds project-local source directories for chromobius and ldpc.
    """
    if project_root is None:
        # src-layout: <repo>/src/concat_decoder/utils/path_setup.py -> repo root is 3 levels up
        project_root = Path(__file__).resolve().parents[3]

    candidates = [
        project_root / "chromobius" / "src",
        project_root / "ldpc_v2_modifiedUF" / "src_python",
        project_root / "color-code-stim" / "src",
    ]
    for path in candidates:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    return project_root
