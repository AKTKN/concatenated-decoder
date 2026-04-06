from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Any, Literal, Sequence
import numpy as np

NoiseModelName = Literal[
    "si1000",
    "uniform_depolarizing",
    "depolarizing",
    "none",
    "code_capacity_depolarizing",
    "code_capacity",
]

CircuitSource = Literal["chromobius", "color-code-stim"]

PlotLerUnit = Literal["per_round", "per_shot", "both"]
PlotXScale = Literal["log", "linear", "both"]
Stage2GraphDecoder = Literal["uf", "mwpm"]
Stage2GraphNonGraphlikePolicy = Literal["reject", "drop"]
RelayBpDecoderVariant = Literal["f32", "f64", "i32", "i64"]
PScale = Literal["log", "linear"]
StatisticalErrorMethod = Literal["wilson"]
BpSchedule = Literal["parallel", "serial"]
DecoderStrategy = Literal[
    "bplsd",
    "relay_bp",
    "chromobius",
    "belief_matching",
    "belief_find",
    "belief_concatmwpm",
    "mwpm",
    "uf",
    "unionfind",
]

ColorSelectionStrategy = Literal["all"] | Sequence[str]


@dataclass(frozen=True)
class CircuitConfig:
    style: str = "superdense_color_code_{basis}"
    diameter: int = 5
    rounds: int | str = 5
    noise_rounds: int | str | None = None
    noise_model: NoiseModelName = "si1000"
    circuit_from: CircuitSource = "chromobius"
    # concatbp-local options. Mapped inside circuit_factory to backend-specific APIs.
    circuit_options: dict[str, Any] = field(default_factory=dict)
    convert_to_cz: bool = True
    editable_extras: dict[str, Any] = field(default_factory=dict)

    def resolved_style(self, basis: str | None = None) -> str:
        if "{basis}" in self.style:
            if basis is None:
                raise ValueError("Circuit style contains '{basis}' but basis was not provided")
            return self.style.format(basis=basis)
        return self.style

    def resolved_rounds(self) -> int:
        r = self.rounds
        if isinstance(r, int):
            if r < 1: raise ValueError("rounds must be >= 1")
            return r

        s = str(r).strip().lower().replace(" ", "")
        if not s: raise ValueError("rounds expression is empty")
        if s.isdigit():
            v = int(s)
            if v < 1: raise ValueError("rounds must be >= 1")
            return v
        if s == "d": return int(self.diameter)

        m = re.fullmatch(r"(\d+)\*?d", s)
        if m: return int(m.group(1)) * int(self.diameter)

        m = re.fullmatch(r"d\*(\d+)", s)
        if m: return int(m.group(1)) * int(self.diameter)

        raise ValueError("Unsupported rounds format. Use int, 'd', '4d', 'd*4', or '4*d'.")

    def resolved_noise_rounds(self) -> int:
        if self.noise_rounds is None:
            return self.resolved_rounds()
        tmp = CircuitConfig(
            style=self.style,
            diameter=self.diameter,
            rounds=self.noise_rounds,
            noise_model=self.noise_model,
            circuit_from=self.circuit_from,
            circuit_options=self.circuit_options,
            convert_to_cz=self.convert_to_cz,
        )
        return tmp.resolved_rounds()


@dataclass(frozen=True)
class RelayBPConfig:
    enabled: bool = False
    decoder_variant: RelayBpDecoderVariant = "f32"
    gamma0: float = 0.1
    pre_iter: int = 80
    num_sets: int = 60
    set_max_iter: int = 60
    gamma_dist_interval: tuple[float, float] = (-0.24, 0.66)
    stop_nconv: int = 1


@dataclass(frozen=True)
class BeliefConcatMWPMConfig:
    # If True, do not run Concat MWPM fallback when full-DEM Relay-BP does not converge.
    # The BP decoding result is used as-is.
    bp_only: bool = False


@dataclass(frozen=True)
class HybridTwoStageConfig:
    enabled: bool = False
    stage2_graph_iter_threshold: int = 12
    stage2_graph_decoder: Stage2GraphDecoder = "uf"
    stage2_graph_target_basis: str = "auto"
    stage2_graph_non_graphlike_policy: Stage2GraphNonGraphlikePolicy = "reject"
    detailed_stats: bool = True
    debug_dump_enabled: bool = False
    debug_dump_dir: str = "concatbp_outputs/hybrid_debug"
    debug_dump_max_shots: int = 8


@dataclass(frozen=True)
class StatisticalErrorConfig:
    enabled: bool = True
    method: StatisticalErrorMethod = "wilson"
    alpha: float = 0.01

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> "StatisticalErrorConfig":
        if data is None:
            data = {}
        return StatisticalErrorConfig(
            enabled=bool(data.get("enabled", True)),
            method=str(data.get("method", "wilson")).strip().lower(),
            alpha=float(data.get("alpha", 0.01)),
        )


@dataclass(frozen=True)
class DecoderConfig:
    strategy: DecoderStrategy = "bplsd"  # Controls the main decoding loop strategy
    colors: Sequence[str] = ("r", "g", "b")
    # When True, drop non-graphlike (3+ detector) error mechanisms in stages that are
    # decoded using MWPM/UF graph decoders. This matches color-code-stim's
    # remove_non_edge_like_errors behavior.
    ignore_hyperedge_error_mechanism: bool = True
    # When True, emit a compact warning if any hyperedge mechanisms are actually
    # dropped while building graphlike DEMs (e.g. for MWPM/UF). The warning includes
    # which color and which stage (stage1/stage2) had mechanisms removed.
    warn_on_hyperedge_drops: bool = True
    # Debug: when Stage-1 BP fails and we fall back to using BP posteriors as priors
    # for graph decoding, print a small summary of posterior LLR/prob values.
    debug_print_bp_posterior: bool = False
    # Debug: print a summary at the end of an experiment task of how often Stage-1
    # Relay-BP failed to converge (belief_* strategies). Counts are weighted by the
    # number of shots, even though decoding caches unique syndromes.
    debug_print_stage1_bp_summary: bool = False
    # Debug: when Stage-1 BP fails to converge, print a small summary of the priors
    # (and derived weights/LLRs) that are passed into MWPM/UF for Stage-1 X/Z graphs.
    debug_print_stage1_bp_failure_priors: bool = False
    # Rate limit for per-failure prior debug prints.
    debug_print_stage1_bp_failure_priors_max_prints: int = 2

    # Union-Find decoder option: when True, request per-cluster stats from ldpc's
    # UnionFindDecoder (decode/decode_batch) and propagate them through concatbp.
    # This does not change decoding results.
    get_cluster_stats: bool = False

    # Which color hypotheses to evaluate in the two-stage concat decoders.
    # - "all": evaluate all colors in `colors` (default; preserves existing behavior)
    # - ["r"], ["r", "g"], ...: evaluate only those colors and select the min-cost solution.
    color_selection_strategy: ColorSelectionStrategy = "all"

    # Post-selection option: when True and 2+ solutions are evaluated, abort a shot if
    # the candidate solutions disagree on logical-operator commutation (observables).
    do_post_selection: bool = False
    bp_method: str = "minimum_sum"
    schedule: BpSchedule = "parallel"
    lsd_method: str = "LSD_0"
    lsd_order: int = 0
    max_iter: int = 30
    ms_scaling_factor: float = 0.75
    relay_bp: RelayBPConfig = field(default_factory=RelayBPConfig)
    belief_concatmwpm: BeliefConcatMWPMConfig = field(default_factory=BeliefConcatMWPMConfig)
    hybrid_two_stage: HybridTwoStageConfig = field(default_factory=HybridTwoStageConfig)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DecoderConfig":
        raw = dict(data)

        # Normalize strategy aliases (backward compatibility).
        # Canonical names used by concatbp:
        # - 'uf' for union-find
        if "strategy" in raw:
            s = str(raw.get("strategy", "bplsd")).strip().lower()
            if s in {"unionfind", "union_find", "union-find", "uf"}:
                raw["strategy"] = "uf"

        # Normalize/validate color selection strategy.
        raw_colors = raw.get("colors", ("r", "g", "b"))
        colors = tuple(str(c).strip().lower() for c in raw_colors)

        css = raw.get("color_selection_strategy", "all")
        if isinstance(css, str):
            css_norm = css.strip().lower()
            if css_norm != "all":
                raise ValueError(
                    "color_selection_strategy must be 'all' or a list of color tokens like ['r','g']"
                )
            raw["color_selection_strategy"] = "all"
        elif isinstance(css, (list, tuple)):
            sel = tuple(str(c).strip().lower() for c in css)
            if not sel:
                raise ValueError("color_selection_strategy list must be non-empty")
            bad = [c for c in sel if c not in colors]
            if bad:
                raise ValueError(
                    f"color_selection_strategy contains invalid color(s) {bad}. Valid: {list(colors)}"
                )
            raw["color_selection_strategy"] = sel
        else:
            raise ValueError(
                "color_selection_strategy must be 'all' or a list of color tokens like ['r','g']"
            )

        raw["do_post_selection"] = bool(raw.get("do_post_selection", False))
        raw["warn_on_hyperedge_drops"] = bool(raw.get("warn_on_hyperedge_drops", True))

        # BP schedule for ldpc.BpLsdDecoder (parallel/serial).
        sched = str(raw.get("schedule", "parallel")).strip().lower()
        if sched not in {"parallel", "serial"}:
            raise ValueError("schedule must be 'parallel' or 'serial'")
        raw["schedule"] = sched

        relay_raw = dict(raw.pop("relay_bp", {}))
        belief_concatmwpm_raw = dict(raw.pop("belief_concatmwpm", {}))
        hybrid_raw = dict(raw.pop("hybrid_two_stage", {}))

        relay = RelayBPConfig(
            enabled=bool(relay_raw.get("enabled", False)),
            decoder_variant=str(relay_raw.get("decoder_variant", "f32")),
            gamma0=float(relay_raw.get("gamma0", 0.1)),
            pre_iter=int(relay_raw.get("pre_iter", 80)),
            num_sets=int(relay_raw.get("num_sets", 60)),
            set_max_iter=int(relay_raw.get("set_max_iter", 60)),
            gamma_dist_interval=tuple(relay_raw.get("gamma_dist_interval", (-0.24, 0.66))),
            stop_nconv=int(relay_raw.get("stop_nconv", 1)),
        )

        belief_concatmwpm = BeliefConcatMWPMConfig(
            bp_only=bool(belief_concatmwpm_raw.get("bp_only", False)),
        )

        hybrid = HybridTwoStageConfig(
            enabled=bool(hybrid_raw.get("enabled", False)),
            stage2_graph_iter_threshold=int(hybrid_raw.get("stage2_graph_iter_threshold", 12)),
            stage2_graph_decoder=str(hybrid_raw.get("stage2_graph_decoder", "uf")),
            stage2_graph_target_basis=str(hybrid_raw.get("stage2_graph_target_basis", "auto")),
            stage2_graph_non_graphlike_policy=str(hybrid_raw.get("stage2_graph_non_graphlike_policy", "reject")),
            detailed_stats=bool(hybrid_raw.get("detailed_stats", False)),
            debug_dump_enabled=bool(hybrid_raw.get("debug_dump_enabled", False)),
            debug_dump_dir=str(hybrid_raw.get("debug_dump_dir", "concatbp_outputs/hybrid_debug")),
            debug_dump_max_shots=int(hybrid_raw.get("debug_dump_max_shots", 8)),
        )
        
        return DecoderConfig(
            **raw,
            relay_bp=relay,
            belief_concatmwpm=belief_concatmwpm,
            hybrid_two_stage=hybrid,
        )


@dataclass(frozen=True)
class ExperimentDetailedStatsConfig:
    """Controls shot-level detailed stats output from the experiment runner."""

    enabled: bool = False
    output_subdir: str = "detailed_stats"
    compression: str = "zstd"

    @staticmethod
    def from_any(value: Any) -> "ExperimentDetailedStatsConfig":
        if value is None:
            return ExperimentDetailedStatsConfig()
        if isinstance(value, bool):
            return ExperimentDetailedStatsConfig(enabled=bool(value))
        if isinstance(value, dict):
            return ExperimentDetailedStatsConfig(
                enabled=bool(value.get("enabled", True)),
                output_subdir=str(value.get("output_subdir", "detailed_stats")),
                compression=str(value.get("compression", "zstd")),
            )
        raise ValueError("detailed_stats must be a bool or dict")


@dataclass(frozen=True)
class ExperimentClusterStatsConfig:
    """Controls shot-level UF cluster stats output from the experiment runner."""

    enabled: bool = False
    output_subdir: str = "cluster_stats"
    compression: str = "zstd"

    @staticmethod
    def from_any(value: Any) -> "ExperimentClusterStatsConfig":
        if value is None:
            return ExperimentClusterStatsConfig()
        if isinstance(value, bool):
            return ExperimentClusterStatsConfig(enabled=bool(value))
        if isinstance(value, dict):
            return ExperimentClusterStatsConfig(
                enabled=bool(value.get("enabled", True)),
                output_subdir=str(value.get("output_subdir", "cluster_stats")),
                compression=str(value.get("compression", "zstd")),
            )
        raise ValueError("cluster_stats must be a bool or dict")


@dataclass(frozen=True)
class ThresholdExperimentConfig:
    circuit_style: str = "superdense_color_code_{basis}"
    circuit_from: CircuitSource = "chromobius"
    # concatbp-local options. Mapped inside circuit_factory to backend-specific APIs.
    circuit_options: dict[str, Any] = field(default_factory=dict)
    # chromobius circuit-generation knob: when True, convert CX->CZ by adding 1Q gates.
    # Default keeps existing concatbp behavior.
    convert_to_cz: bool = True
    # chromobius circuit-generation extras forwarded into clorco make_circuit.
    editable_extras: dict[str, Any] = field(default_factory=dict)
    basis: str = "Z"
    distances: Sequence[int] = (3, 5, 7)
    p_min: float = 0.01
    p_max: float = 0.1
    num_p: int = 5
    p_values: Sequence[float] | None = None
    p_scale: PScale = "log"
    noise_model: str = "depolarizing"
    rounds: int | str = 1
    shots: int = 10000
    shots_per_batch: int = 1000
    seed: int = 0
    workers: int | None = None
    verbose: bool = True
    plot_ler_unit: PlotLerUnit = "per_round"
    plot_x_scale: PlotXScale = "log"
    output_dir: str = "concatbp_outputs/threshold"
    resume: bool = True
    statistics: StatisticalErrorConfig = field(default_factory=StatisticalErrorConfig)

    # When True and circuit_from='color-code-stim', run comparative decoding across all
    # logical classes (sets observable-detector bits and chooses the min-weight class).
    comparative_decoding: bool = False

    # Shot-level detailed stats output (separate from decoder.hybrid_two_stage.detailed_stats).
    detailed_stats: ExperimentDetailedStatsConfig = field(default_factory=ExperimentDetailedStatsConfig)

    # Shot-level UF cluster stats output (separate from detailed_stats).
    cluster_stats: ExperimentClusterStatsConfig = field(default_factory=ExperimentClusterStatsConfig)

    def resolved_p_values(self) -> np.ndarray:
        if self.p_values is not None:
            return np.asarray(self.p_values, dtype=float)
        scale = str(self.p_scale).strip().lower()
        if scale in {"log", "geom", "geomspace"}:
            return np.geomspace(self.p_max, self.p_min, self.num_p)
        if scale in {"linear", "lin", "linspace"}:
            return np.linspace(self.p_max, self.p_min, self.num_p)
        raise ValueError(f"Unknown p_scale: {self.p_scale}. Use 'log' or 'linear'.")

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ThresholdExperimentConfig":
        raw = dict(data)
        stats_raw = raw.pop("statistics", raw.pop("stats", None))
        stats = StatisticalErrorConfig.from_dict(stats_raw)

        detailed_raw = raw.pop("detailed_stats", None)
        detailed = ExperimentDetailedStatsConfig.from_any(detailed_raw)

        cluster_raw = raw.pop("cluster_stats", None)
        cluster = ExperimentClusterStatsConfig.from_any(cluster_raw)

        # Allow legacy naming variants if present.
        if "comparative" in raw and "comparative_decoding" not in raw:
            raw["comparative_decoding"] = bool(raw.pop("comparative"))

        return ThresholdExperimentConfig(**raw, statistics=stats, detailed_stats=detailed, cluster_stats=cluster)

@dataclass(frozen=True)
class ThresholdPoint:
    distance: int
    # Second distance parameter used by some color-code-stim layouts (e.g. rec/growing).
    # When unset / not applicable, this is None.
    d2: int | None
    p: float
    style: str
    noise_model: str
    ler_x: float
    ler_z: float
    ler_x_ci_low: float
    ler_x_ci_high: float
    ler_z_ci_low: float
    ler_z_ci_high: float
    shots: int
    rounds_spec: str
    requested_rounds: int
    effective_rounds: int

    # Post-selection metrics (optional).
    abort_count: int = 0
    abort_rate: float = 0.0
    postselected_ler_x: float = float("nan")
    postselected_ler_z: float = float("nan")

@dataclass(frozen=True)
class _ThresholdTask:
    distance: int
    d2: int | None
    p: float
    seed: int

@dataclass(frozen=True)
class _ThresholdTaskResult:
    point: ThresholdPoint
    detailed_stats: list[dict[str, float | int | str]]