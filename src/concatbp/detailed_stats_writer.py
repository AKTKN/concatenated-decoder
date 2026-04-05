from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ._debug_io import safe_token


def make_detailed_stats_filename(
    *,
    circuit_style: str,
    layout: str,
    basis: str,
    d: int,
    d2: int | None,
    p: float,
    noise_model: str,
) -> str:
    """Builds the required filename for shot-level detailed stats parquet output."""
    p_tok = format(float(p), ".12g")
    parts = {
        "circuit_style": circuit_style,
        "layout": layout,
        "basis": basis,
        "d": str(int(d)),
        "d2": str(d2),
        "p": p_tok,
        "noise": noise_model,
    }
    joined = ",".join(f"{k}={safe_token(v)}" for k, v in parts.items())
    return f"{joined}.parquet"


def _require_pyarrow():
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to write shot-level detailed stats. "
            "Install it with `pip install pyarrow`."
        ) from exc


def shot_stats_columns() -> list[tuple[str, np.dtype]]:
    """Column definitions for shot-level detailed stats."""
    return [
        ("shot_index", np.dtype(np.int64)),
        ("logical_error", np.dtype(np.bool_)),
        ("num_errors", np.dtype(np.int64)),
        ("decode_time_ms", np.dtype(np.float64)),
        ("iteration", np.dtype(np.float64)),
        ("r_weight", np.dtype(np.float64)),
        ("g_weight", np.dtype(np.float64)),
        ("b_weight", np.dtype(np.float64)),
        ("s1_rweight", np.dtype(np.float64)),
        ("s1_gweight", np.dtype(np.float64)),
        ("s1_bweight", np.dtype(np.float64)),
        ("logical_gap", np.dtype(np.float64)),
    ]


def _shot_stats_schema():
    _require_pyarrow()
    import pyarrow as pa

    # Use explicit types for stable output schema.
    fields = [
        pa.field("shot_index", pa.int64()),
        pa.field("logical_error", pa.bool_()),
        pa.field("num_errors", pa.int64()),
        pa.field("decode_time_ms", pa.float64()),
        pa.field("iteration", pa.float64()),
        pa.field("r_weight", pa.float64()),
        pa.field("g_weight", pa.float64()),
        pa.field("b_weight", pa.float64()),
        pa.field("s1_rweight", pa.float64()),
        pa.field("s1_gweight", pa.float64()),
        pa.field("s1_bweight", pa.float64()),
        pa.field("logical_gap", pa.float64()),
    ]
    return pa.schema(fields)


@dataclass
class ShotStatsParquetWriter:
    """Streaming Parquet writer for shot-level detailed stats.

    Writes to a temporary file and atomically replaces the final output path on close.
    """

    final_path: Path
    compression: str = "zstd"

    def __post_init__(self) -> None:
        _require_pyarrow()
        import pyarrow.parquet as pq

        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f".{self.final_path.name}.tmp_pid={os.getpid()}"
        self._tmp_path = self.final_path.with_name(tmp_name)

        self._schema = _shot_stats_schema()
        self._writer = pq.ParquetWriter(
            where=str(self._tmp_path),
            schema=self._schema,
            compression=str(self.compression),
        )
        self._closed = False

    def write_batch(
        self,
        *,
        shot_index: np.ndarray,
        logical_error: np.ndarray,
        num_errors: np.ndarray,
        decode_time_ms: np.ndarray,
        iteration: np.ndarray,
        r_weight: np.ndarray,
        g_weight: np.ndarray,
        b_weight: np.ndarray,
        s1_rweight: np.ndarray,
        s1_gweight: np.ndarray,
        s1_bweight: np.ndarray,
        logical_gap: np.ndarray,
    ) -> None:
        """Write a batch of rows.

        All arrays must be 1D and have the same length.
        """
        _require_pyarrow()
        import pyarrow as pa

        cols: dict[str, np.ndarray] = {
            "shot_index": np.asarray(shot_index, dtype=np.int64).ravel(),
            "logical_error": np.asarray(logical_error, dtype=np.bool_).ravel(),
            "num_errors": np.asarray(num_errors, dtype=np.int64).ravel(),
            "decode_time_ms": np.asarray(decode_time_ms, dtype=np.float64).ravel(),
            "iteration": np.asarray(iteration, dtype=np.float64).ravel(),
            "r_weight": np.asarray(r_weight, dtype=np.float64).ravel(),
            "g_weight": np.asarray(g_weight, dtype=np.float64).ravel(),
            "b_weight": np.asarray(b_weight, dtype=np.float64).ravel(),
            "s1_rweight": np.asarray(s1_rweight, dtype=np.float64).ravel(),
            "s1_gweight": np.asarray(s1_gweight, dtype=np.float64).ravel(),
            "s1_bweight": np.asarray(s1_bweight, dtype=np.float64).ravel(),
            "logical_gap": np.asarray(logical_gap, dtype=np.float64).ravel(),
        }

        lens = {v.shape[0] for v in cols.values()}
        if len(lens) != 1:
            raise ValueError(f"Length mismatch in shot stats columns: { {k: v.shape for k, v in cols.items()} }")

        arrays = [pa.array(cols[name], type=self._schema.field(name).type) for name in self._schema.names]
        table = pa.Table.from_arrays(arrays, names=self._schema.names)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._closed:
            return
        self._writer.close()
        os.replace(self._tmp_path, self.final_path)
        self._closed = True

    def __enter__(self) -> "ShotStatsParquetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # If an exception happens, keep the temp file for inspection.
        if exc_type is None:
            self.close()
        else:
            try:
                self._writer.close()
            except Exception:
                pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def summarize_parquet_files(paths: Iterable[Path]) -> str:
    """Small helper for logging/debug prints."""
    return "\n".join(f"- {p}" for p in paths)
