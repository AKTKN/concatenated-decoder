from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ._debug_io import safe_token


def make_cluster_stats_filename(
    *,
    circuit_style: str,
    layout: str,
    basis: str,
    d: int,
    d2: int | None,
    p: float,
    noise_model: str,
) -> str:
    """Builds the filename for shot-level UF cluster stats parquet output."""
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
            "pyarrow is required to write cluster_stats parquet output. "
            "Install it with `pip install pyarrow`."
        ) from exc


def _cluster_stats_schema():
    _require_pyarrow()
    import pyarrow as pa

    # Use explicit types for stable output schema.
    fields = [
        pa.field("shot_index", pa.int64()),
        pa.field("logical_error", pa.bool_()),
        pa.field("selected_color", pa.int8()),
    ]

    # Spec-coded color columns are emitted for all of r/b/g.
    colors = ("r", "b", "g")
    stages = ("s1", "s2")

    for stage in stages:
        for c in colors:
            fields.append(pa.field(f"{stage}_{c}_max_cluster_syndrome_count", pa.int32()))

    for stage in stages:
        for c in colors:
            fields.append(pa.field(f"{stage}_{c}_max_cluster_llr", pa.float64()))

    return pa.schema(fields)


@dataclass
class ClusterStatsParquetWriter:
    """Streaming Parquet writer for UF cluster stats (shot-level)."""

    final_path: Path
    compression: str = "zstd"

    def __post_init__(self) -> None:
        _require_pyarrow()
        import pyarrow.parquet as pq

        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f".{self.final_path.name}.tmp_pid={os.getpid()}"
        self._tmp_path = self.final_path.with_name(tmp_name)

        self._schema = _cluster_stats_schema()
        self._writer = pq.ParquetWriter(
            where=str(self._tmp_path),
            schema=self._schema,
            compression=str(self.compression),
        )
        self._closed = False

    @property
    def schema_names(self) -> list[str]:
        return list(self._schema.names)

    def write_batch(self, cols: dict[str, np.ndarray]) -> None:
        """Write a batch of rows.

        `cols` must contain every schema column name as a key.
        All arrays must be 1D and have the same length.
        """
        _require_pyarrow()
        import pyarrow as pa

        missing = [n for n in self._schema.names if n not in cols]
        if missing:
            raise ValueError(f"Missing cluster_stats columns: {missing}")

        casted: dict[str, np.ndarray] = {}
        for name in self._schema.names:
            arr = np.asarray(cols[name]).ravel()
            field_type = self._schema.field(name).type
            if name == "shot_index":
                arr = arr.astype(np.int64, copy=False)
            elif name == "logical_error":
                arr = arr.astype(np.bool_, copy=False)
            elif name == "selected_color":
                arr = arr.astype(np.int8, copy=False)
            elif name.endswith("_max_cluster_syndrome_count"):
                arr = arr.astype(np.int32, copy=False)
            elif name.endswith("_max_cluster_llr"):
                arr = arr.astype(np.float64, copy=False)
            casted[name] = arr

        lens = {v.shape[0] for v in casted.values()}
        if len(lens) != 1:
            raise ValueError(f"Length mismatch in cluster_stats columns: { {k: v.shape for k, v in casted.items()} }")

        arrays = [pa.array(casted[name], type=self._schema.field(name).type) for name in self._schema.names]
        table = pa.Table.from_arrays(arrays, names=self._schema.names)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._closed:
            return
        self._writer.close()
        os.replace(self._tmp_path, self.final_path)
        self._closed = True

    def __enter__(self) -> "ClusterStatsParquetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self._writer.close()
            except Exception:
                pass


def summarize_parquet_files(paths: Iterable[Path]) -> str:
    return "\n".join(f"- {p}" for p in paths)
