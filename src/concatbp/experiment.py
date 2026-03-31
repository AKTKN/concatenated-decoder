from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import stim
import numpy as np
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .dem_model import BinaryErrorModel

from .circuit_factory import build_circuit
from .config import CircuitConfig, DecoderConfig, ThresholdExperimentConfig, ThresholdPoint, _ThresholdTask, _ThresholdTaskResult
from .decoders import (
    ConcatBPLSDDecoder,
    ConcatRelayBPDecoder,
    ChromobiusDecoder,
    ConcatBeliefMatchingDecoder,
    ConcatBeliefFindDecoder,
    ConcatBeliefConcatMWPMDecoder,
    ConcatMatchingDecoder,
    ConcatFindDecoder,
)
from .dem_model import parse_detector_metadata, parse_stim_dem
from .stats_io import (
    write_detailed_stats_parquet, load_existing_points, merge_existing_rows, 
    write_threshold_csv, write_meta
)
from .visualization import plot_thresholds
from .simulation_utils import compute_failure_stats, scale_interval
from .comparative_decoding import decode_batch_comparative, get_obs_detector_ids_by_obs_index
from .detailed_stats_writer import ShotStatsParquetWriter, make_detailed_stats_filename
from ._debug_io import dump_stim_circuit


def _resolve_task_d2(cfg: ThresholdExperimentConfig, *, distance: int) -> int | None:
    """Resolve the concrete d2 value for a given distance.

    Only applies to the color-code-stim backend (layouts like rec/growing). For other
    backends or when not configured, returns None.
    """
    if str(cfg.circuit_from) != "color-code-stim":
        return None
    opts = dict(getattr(cfg, "circuit_options", {}) or {})
    if "d2" not in opts:
        return None
    raw = opts.get("d2")
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(f"circuit_options.d2 must be an int or str, got bool: {raw!r}")
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, str):
        tok = raw.strip().lower().replace(" ", "")
        if tok in {"d", "distance"}:
            return int(distance)
        if tok.startswith("d+") and tok[2:].isdigit():
            return int(distance) + int(tok[2:])
        raise ValueError(
            "Unsupported d2 string format. Use an integer, 'd'/'distance', or 'd+N'. "
            f"Got d2={raw!r}"
        )
    raise ValueError(
        "circuit_options.d2 must be an int or str (e.g. 6, 'd', 'd+2'). "
        f"Got type={type(raw).__name__} value={raw!r}"
    )

def _format_binary_model_for_debug(name: str, model: "BinaryErrorModel") -> str:
    lines = [
        f"[{name}]",
        f"num_detectors={model.num_detectors}",
        f"num_observables={model.num_observables}",
        f"num_mechanisms={model.num_errors}",
    ]
    flip_indices = [i for i, o in enumerate(model.observable_targets) if len(o) > 0]
    lines.append(f"observable_flipping_mechanism_indices={flip_indices}")
    for i, (p, d, o) in enumerate(zip(model.probabilities, model.detector_targets, model.observable_targets)):
        lines.append(f"idx={i} p={float(p):.12g} det={list(map(int, d))} obs={list(map(int, o))} graphlike={len(d) <= 2}")
    return "\n".join(lines) + "\n"

def _create_decoder(
    strategy: str,
    base_model: "BinaryErrorModel",
    dem: stim.DetectorErrorModel | None,
    metadata: Any,
    decoder_cfg: DecoderConfig,
    debug_base_dir: Path | None,
    debug_run_tag: str | None,
):
    """Factory function to instantiate the correct Decoder strategy based on config."""
    strategy = str(strategy).strip().lower()
    if strategy in {"unionfind", "union_find", "union-find"}:
        strategy = "uf"

    kwargs = {}
    if debug_base_dir is not None:
        kwargs["debug_output_dir"] = str(debug_base_dir)
    if debug_run_tag is not None:
        kwargs["debug_run_tag"] = debug_run_tag

    # Pass the basis information required for graph separation strategies
    if strategy in ["belief_matching", "belief_find", "belief_concatmwpm", "mwpm", "uf"]:
        if not hasattr(metadata, "detector_basis"):
            raise ValueError(f"Strategy '{strategy}' requires detector_basis in metadata.")
        kwargs["detector_basis_by_id"] = metadata.detector_basis

    if strategy == "chromobius":
        if dem is None:
            raise ValueError("chromobius strategy requires dem")
        return ChromobiusDecoder(dem)
    elif strategy == "bplsd":
        return ConcatBPLSDDecoder(base_model, metadata.detector_ids_by_color, decoder_cfg, **kwargs)
    elif strategy == "relay_bp":
        return ConcatRelayBPDecoder(base_model, metadata.detector_ids_by_color, decoder_cfg, **kwargs)
    elif strategy == "belief_matching":
        return ConcatBeliefMatchingDecoder(base_model, metadata.detector_ids_by_color, decoder_cfg, **kwargs)
    elif strategy == "belief_find":
        return ConcatBeliefFindDecoder(base_model, metadata.detector_ids_by_color, decoder_cfg, **kwargs)
    elif strategy == "belief_concatmwpm":
        return ConcatBeliefConcatMWPMDecoder(base_model, metadata.detector_ids_by_color, decoder_cfg, **kwargs)
    elif strategy == "mwpm":
        return ConcatMatchingDecoder(base_model, metadata.detector_ids_by_color, decoder_cfg, **kwargs)
    elif strategy == "uf":
        return ConcatFindDecoder(base_model, metadata.detector_ids_by_color, decoder_cfg, **kwargs)
    else:
        raise ValueError(f"Unknown decoding strategy: {strategy}")

def _run_threshold_task(
    task: _ThresholdTask,
    cfg: ThresholdExperimentConfig,
    decoder_cfg: DecoderConfig,
) -> _ThresholdTaskResult:
    if str(decoder_cfg.strategy).strip().lower() == "chromobius" and str(cfg.circuit_from) != "chromobius":
        raise ValueError("decoder.strategy='chromobius' requires experiment.circuit_from='chromobius'")

    comparative = bool(getattr(cfg, "comparative_decoding", False))
    detailed_stats_cfg = getattr(cfg, "detailed_stats", None)
    detailed_enabled = bool(getattr(detailed_stats_cfg, "enabled", False))

    strategy_norm = str(decoder_cfg.strategy).strip().lower()
    if comparative:
        if str(cfg.circuit_from) != "color-code-stim":
            raise ValueError("experiment.comparative_decoding requires experiment.circuit_from='color-code-stim'")
        if strategy_norm in {"chromobius", "belief_concatmwpm"}:
            raise ValueError("comparative_decoding is not supported for decoder.strategy='chromobius' or 'belief_concatmwpm'")
        if bool(getattr(decoder_cfg, "do_post_selection", False)):
            raise ValueError("comparative_decoding does not support decoder.do_post_selection")

    if detailed_enabled:
        # Shot-level detailed stats require per-color candidate weights and stage-1 weights.
        if strategy_norm in {"chromobius", "belief_concatmwpm"}:
            raise ValueError(
                "experiment.detailed_stats is not supported for decoder.strategy='chromobius' or 'belief_concatmwpm'"
            )

    circuit_cfg = CircuitConfig(
        style=cfg.circuit_style,
        diameter=int(task.distance),
        rounds=cfg.rounds,
        noise_model=cfg.noise_model,
        circuit_from=cfg.circuit_from,
        circuit_options=dict(cfg.circuit_options),
        convert_to_cz=bool(getattr(cfg, "convert_to_cz", True)),
        editable_extras=dict(getattr(cfg, "editable_extras", {})),
    )

    if comparative:
        # Forward to circuit_factory -> color-code-stim ColorCode(comparative_decoding=True)
        circuit_cfg = CircuitConfig(
            **{
                **circuit_cfg.__dict__,
                "circuit_options": {**dict(circuit_cfg.circuit_options), "enable_comparative_decoding": True},
            }
        )
    built = build_circuit(circuit_cfg, basis=cfg.basis, p=float(task.p))
    dem = built.circuit.detector_error_model(decompose_errors=False)
    base_model = parse_stim_dem(dem)
    metadata = parse_detector_metadata(built.circuit)

    debug_run_tag = None
    debug_base_dir = None
    if decoder_cfg.hybrid_two_stage.debug_dump_enabled:
        debug_base_dir = Path(decoder_cfg.hybrid_two_stage.debug_dump_dir)
        debug_run_tag = (
            f"threshold_d={int(task.distance)}_d2={task.d2}_p={float(task.p):.9g}_seed={int(task.seed)}_pid={os.getpid()}"
        )
        task_dir = debug_base_dir / debug_run_tag
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "00_circuit.stim").write_text(str(built.circuit), encoding="utf-8")
        (task_dir / "01_dem.dem").write_text(str(dem), encoding="utf-8")
        (task_dir / "02_base_model.txt").write_text(_format_binary_model_for_debug("base_model", base_model), encoding="utf-8")

    # Instantiate decoder dynamically
    decoder = _create_decoder(
        strategy=decoder_cfg.strategy,
        base_model=base_model,
        dem=dem,
        metadata=metadata,
        decoder_cfg=decoder_cfg,
        debug_base_dir=debug_base_dir,
        debug_run_tag=debug_run_tag,
    )

    sampler = built.circuit.compile_detector_sampler(seed=int(task.seed))
    mismatch_count = 0
    post_mismatch_count = 0
    abort_count = 0
    post_shots = 0
    processed = 0
    detailed_rows: list[dict[str, float | int | str]] = []
    batch_size = max(1, int(cfg.shots_per_batch))

    # Reuse a buffer for XOR mismatch computation to reduce allocations.
    # (Does not change the underlying logic/results.)
    num_obs = int(base_model.num_observables)
    xor_buf = np.empty((batch_size, num_obs), dtype=np.uint8)

    # comparative decoding needs mapping from observable index -> detector id.
    obs_det_ids_by_obs: list[int] | None = None
    if comparative:
        obs_det_ids_by_obs = get_obs_detector_ids_by_obs_index(built.circuit)

    # Optional shot-level parquet writer.
    writer: ShotStatsParquetWriter | None = None
    if detailed_enabled:
        out_dir = Path(cfg.output_dir)
        layout = str(cfg.circuit_options.get("layout", ""))
        fname = make_detailed_stats_filename(
            circuit_style=str(built.style),
            layout=layout,
            basis=str(cfg.basis),
            d=int(task.distance),
            d2=task.d2,
            p=float(task.p),
            noise_model=str(cfg.noise_model),
        )
        writer = ShotStatsParquetWriter(
            final_path=out_dir / str(getattr(detailed_stats_cfg, "output_subdir", "detailed_stats")) / fname,
            compression=str(getattr(detailed_stats_cfg, "compression", "zstd")),
        )

    try:
        while processed < cfg.shots:
            n = min(batch_size, cfg.shots - processed)
            det_outcomes, obs_true = sampler.sample(shots=n, separate_observables=True, bit_packed=False)

            obs_true_u8 = np.asarray(obs_true, dtype=np.uint8)

            if comparative:
                assert obs_det_ids_by_obs is not None
                comp_res = decode_batch_comparative(
                    decoder=decoder,
                    detector_outcomes=det_outcomes,
                    obs_detector_ids_by_obs_index=obs_det_ids_by_obs,
                    colors=("r", "g", "b"),
                )
                obs_pred_u8 = np.asarray(comp_res.obs_prediction, dtype=np.uint8)
                decode_time_ms = np.asarray(comp_res.decode_time_ms, dtype=np.float64).reshape(-1)[:n]
                logical_gap = np.asarray(comp_res.logical_gap, dtype=np.float64).reshape(-1)[:n]
                per_color_costs = comp_res.candidate_costs
                per_color_s1 = comp_res.candidate_stage1_costs
            else:
                result = decoder.decode_batch(det_outcomes)
                obs_pred_u8 = np.asarray(result.obs_prediction, dtype=np.uint8)
                decode_time_ms = (
                    np.asarray(result.decode_time_ms, dtype=np.float64).reshape(-1)[:n]
                    if result.decode_time_ms is not None
                    else np.full(n, np.nan, dtype=np.float64)
                )
                logical_gap = np.full(n, np.nan, dtype=np.float64)
                per_color_costs = result.candidate_costs
                if result.candidate_stage1_costs is None:
                    raise ValueError("Decoder did not return candidate_stage1_costs")
                per_color_s1 = result.candidate_stage1_costs

            np.bitwise_xor(obs_true_u8, obs_pred_u8, out=xor_buf[:n, :])
            mismatch_flags = np.any(xor_buf[:n, :], axis=1)
            mismatch_count += int(np.count_nonzero(mismatch_flags))

            # Optional post-selection.
            # chromobius strategy intentionally ignores post-selection.
            do_postselect = (
                bool(getattr(decoder_cfg, "do_post_selection", False))
                and str(decoder_cfg.strategy).strip().lower() != "chromobius"
            )
            if do_postselect:
                if result.abort_mask is None:
                    # No abort information => no shots aborted.
                    post_shots += int(n)
                    post_mismatch_count += int(np.count_nonzero(mismatch_flags))
                else:
                    abort_flags = np.asarray(result.abort_mask, dtype=bool).reshape(-1)[:n]
                    abort_batch = int(np.count_nonzero(abort_flags))
                    abort_count += abort_batch
                    survived = int(n - abort_batch)
                    post_shots += survived
                    if survived:
                        post_mismatch_count += int(np.count_nonzero(mismatch_flags & (~abort_flags)))

            # Shot-level detailed stats (streaming to parquet).
            if writer is not None:
                expected = ("r", "g", "b")
                for c in expected:
                    if c not in per_color_costs or c not in per_color_s1:
                        raise ValueError(f"Decoder missing per-color stats for color={c!r}")

                r_w = np.asarray(per_color_costs["r"], dtype=np.float64).reshape(-1)[:n]
                g_w = np.asarray(per_color_costs["g"], dtype=np.float64).reshape(-1)[:n]
                b_w = np.asarray(per_color_costs["b"], dtype=np.float64).reshape(-1)[:n]
                s1_r = np.asarray(per_color_s1["r"], dtype=np.float64).reshape(-1)[:n]
                s1_g = np.asarray(per_color_s1["g"], dtype=np.float64).reshape(-1)[:n]
                s1_b = np.asarray(per_color_s1["b"], dtype=np.float64).reshape(-1)[:n]

                if comparative:
                    # In comparative decoding, weights are fixed to the chosen logical class;
                    # sign all colors based on the final logical prediction.
                    sign = np.where(mismatch_flags, -1.0, 1.0).astype(np.float64, copy=False)
                    r_w = r_w * sign
                    g_w = g_w * sign
                    b_w = b_w * sign
                else:
                    # In non-comparative decoding, sign each color based on that color's
                    # candidate observable prediction.
                    if result.candidate_obs is None:
                        raise ValueError("Decoder did not return candidate_obs")
                    sign_r = np.where(
                        np.any(result.candidate_obs["r"][:n, :] != obs_true_u8[:n, :], axis=1),
                        -1.0,
                        1.0,
                    )
                    sign_g = np.where(
                        np.any(result.candidate_obs["g"][:n, :] != obs_true_u8[:n, :], axis=1),
                        -1.0,
                        1.0,
                    )
                    sign_b = np.where(
                        np.any(result.candidate_obs["b"][:n, :] != obs_true_u8[:n, :], axis=1),
                        -1.0,
                        1.0,
                    )
                    r_w = r_w * sign_r
                    g_w = g_w * sign_g
                    b_w = b_w * sign_b

                writer.write_batch(
                    shot_index=(processed + np.arange(n, dtype=np.int64)),
                    decode_time_ms=decode_time_ms,
                    r_weight=r_w,
                    g_weight=g_w,
                    b_weight=b_w,
                    s1_rweight=s1_r,
                    s1_gweight=s1_g,
                    s1_bweight=s1_b,
                    logical_gap=logical_gap,
                )

            processed += n
    finally:
        if writer is not None:
            writer.close()

    if bool(getattr(decoder_cfg, "debug_print_stage1_bp_summary", False)) and hasattr(decoder, "print_stage1_bp_summary"):
        tag = (
            f"threshold d={int(task.distance)} basis={str(cfg.basis)} p={float(task.p):.9g} "
            f"shots={int(cfg.shots)} seed={int(task.seed)} pid={os.getpid()}"
        )
        try:
            decoder.print_stage1_bp_summary(tag=tag)
        except Exception as exc:
            print(f"[concatbp][stage1_bp_nonconvergence_summary] failed to print summary: {exc}")

    rounds_effective = max(1, int(built.rounds_effective))
    ler = mismatch_count / float(cfg.shots)
    ler /= float(rounds_effective)

    post_ler = float("nan")
    abort_rate = 0.0
    if bool(getattr(decoder_cfg, "do_post_selection", False)) and str(decoder_cfg.strategy).strip().lower() != "chromobius":
        abort_rate = abort_count / float(cfg.shots)
        if post_shots > 0:
            post_ler = (post_mismatch_count / float(post_shots)) / float(rounds_effective)

    stats = compute_failure_stats(mismatch_count, int(cfg.shots), cfg.statistics)
    ci_low_round, ci_high_round = scale_interval(stats.ci_low, stats.ci_high, 1.0 / float(rounds_effective))
    ci_low_round = float(np.asarray(ci_low_round).reshape(-1)[0])
    ci_high_round = float(np.asarray(ci_high_round).reshape(-1)[0])

    if cfg.basis == "X":
        ler_x, ler_z = float(ler), float("nan")
        ler_x_ci_low, ler_x_ci_high = ci_low_round, ci_high_round
        ler_z_ci_low, ler_z_ci_high = float("nan"), float("nan")
    else:
        ler_x, ler_z = float("nan"), float(ler)
        ler_x_ci_low, ler_x_ci_high = float("nan"), float("nan")
        ler_z_ci_low, ler_z_ci_high = ci_low_round, ci_high_round

    if cfg.basis == "X":
        post_ler_x, post_ler_z = float(post_ler), float("nan")
    else:
        post_ler_x, post_ler_z = float("nan"), float(post_ler)

    point = ThresholdPoint(
        distance=int(task.distance), d2=task.d2, p=float(task.p), style=built.style, noise_model=built.noise_model,
        ler_x=ler_x, ler_z=ler_z,
        ler_x_ci_low=ler_x_ci_low, ler_x_ci_high=ler_x_ci_high,
        ler_z_ci_low=ler_z_ci_low, ler_z_ci_high=ler_z_ci_high,
        shots=int(cfg.shots), rounds_spec=str(built.rounds_spec),
        requested_rounds=int(built.rounds_requested), effective_rounds=int(built.rounds_effective),
        abort_count=int(abort_count),
        abort_rate=float(abort_rate),
        postselected_ler_x=float(post_ler_x),
        postselected_ler_z=float(post_ler_z),
    )
    return _ThresholdTaskResult(point=point, detailed_stats=detailed_rows)

def run_threshold_experiment(
    cfg: ThresholdExperimentConfig,
    decoder_cfg: DecoderConfig,
) -> list[ThresholdPoint]:
    if str(decoder_cfg.strategy).strip().lower() == "chromobius" and str(cfg.circuit_from) != "chromobius":
        raise ValueError("decoder.strategy='chromobius' requires experiment.circuit_from='chromobius'")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_vals = cfg.resolved_p_values()
    rows: list[ThresholdPoint] = []
    detailed_stats_rows: list[dict[str, float | int | str]] = []
    circuits_dir = Path("concatbp_circuits")
    dumped: set[str] = set()
    completed = load_existing_points(out_dir / "threshold_results.csv") if cfg.resume else set()

    tasks: list[_ThresholdTask] = []
    seed_cursor = int(cfg.seed)
    for d in cfg.distances:
        d2 = _resolve_task_d2(cfg, distance=int(d))
        for p in p_vals:
            if (int(d), d2, float(p)) in completed:
                continue
            tasks.append(_ThresholdTask(distance=int(d), d2=d2, p=float(p), seed=seed_cursor))
            seed_cursor += 1

    for task in tasks:
        circuit_cfg = CircuitConfig(
            style=cfg.circuit_style,
            diameter=int(task.distance),
            rounds=cfg.rounds,
            noise_model=cfg.noise_model,
            circuit_from=cfg.circuit_from,
            circuit_options=dict(cfg.circuit_options),
            convert_to_cz=bool(getattr(cfg, "convert_to_cz", True)),
            editable_extras=dict(getattr(cfg, "editable_extras", {})),
        )
        if bool(getattr(cfg, "comparative_decoding", False)):
            circuit_cfg = CircuitConfig(
                **{
                    **circuit_cfg.__dict__,
                    "circuit_options": {**dict(circuit_cfg.circuit_options), "enable_comparative_decoding": True},
                }
            )
        built = build_circuit(circuit_cfg, basis=cfg.basis, p=float(task.p))
        layout = str(cfg.circuit_options.get("layout", ""))
        dump_name = (
            f"threshold_style={built.style},layout={layout},basis={cfg.basis},d={task.distance},d2={task.d2},"
            f"p={float(task.p):.6g},noise={cfg.noise_model},r={built.rounds_effective}"
        )
        if dump_name not in dumped:
            dump_stim_circuit(built.circuit, out_dir=circuits_dir, name=dump_name)
            dumped.add(dump_name)

    workers = cfg.workers
    if workers == 1:
        for i, task in enumerate(tasks, start=1):
            task_result = _run_threshold_task(task, cfg, decoder_cfg)
            rows.append(task_result.point)
            detailed_stats_rows.extend(task_result.detailed_stats)
            if cfg.verbose: print(f"[{i}/{len(tasks)}] d={task.distance} p={task.p:.6g}")
    else:
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
            futures = {executor.submit(_run_threshold_task, task, cfg, decoder_cfg): task for task in tasks}
            done = 0
            for future in as_completed(futures):
                task_result = future.result()
                rows.append(task_result.point)
                detailed_stats_rows.extend(task_result.detailed_stats)
                done += 1
                if cfg.verbose:
                    t = futures[future]
                    print(f"[{done}/{len(tasks)}] d={t.distance} p={t.p:.6g}")

    all_rows = merge_existing_rows(out_dir / "threshold_results.csv", rows)
    write_threshold_csv(all_rows, out_dir / "threshold_results.csv")
    
    if detailed_stats_rows:
        detailed_stats_rows.sort(key=lambda r: (int(r["distance"]), float(r["p"]), int(r.get("shot_index", 0)), str(r.get("color", ""))))
        write_detailed_stats_parquet(detailed_stats_rows, out_dir / "detailed_stats.parquet")
        
    plot_thresholds(all_rows, cfg.basis, out_dir, cfg.plot_ler_unit, cfg.plot_x_scale)
    write_meta(cfg, decoder_cfg, out_dir / "threshold_config.txt")
    
    return all_rows