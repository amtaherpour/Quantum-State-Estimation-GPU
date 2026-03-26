from __future__ import annotations

import copy
import csv
import json
import math
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import ExperimentConfig, LossConfig, SUPPORTED_LOSSES
from experiments import (
    ExperimentRunResult,
    list_available_experiments,
    make_named_experiment,
    run_configured_experiment,
)


# ============================================================
# Small helpers
# ============================================================

def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _ensure_positive_float(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _ensure_nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _ensure_nonempty_string(value: str, name: str) -> str:
    value = str(value).strip()
    if len(value) == 0:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _mkdir_if_needed(path: Optional[str]) -> None:
    if path is None or len(str(path).strip()) == 0:
        return
    os.makedirs(path, exist_ok=True)


def _deepcopy_config(cfg: ExperimentConfig) -> ExperimentConfig:
    return copy.deepcopy(cfg)


def _config_from_input(config_or_name: Union[str, ExperimentConfig]) -> ExperimentConfig:
    if isinstance(config_or_name, ExperimentConfig):
        return _deepcopy_config(config_or_name)
    name = _ensure_nonempty_string(config_or_name, "config_or_name")
    return make_named_experiment(name)


def _maybe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def _detach_to_cpu(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def _to_numpy_1d(x: Any) -> np.ndarray:
    x = _detach_to_cpu(x)
    if isinstance(x, torch.Tensor):
        if torch.is_complex(x):
            raise ValueError("Expected a real-valued array for plotting.")
        return x.reshape(-1).numpy()
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError("Expected a real-valued array for plotting.")
    return arr.reshape(-1)


def _to_serializable(obj: Any) -> Any:
    """
    Recursively convert nested objects into JSON-serializable objects.
    """
    obj = _detach_to_cpu(obj)

    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            value = obj.item()
            if isinstance(value, complex):
                return {"real": float(value.real), "imag": float(value.imag)}
            return value
        if torch.is_complex(obj):
            return {
                "real": obj.real.tolist(),
                "imag": obj.imag.tolist(),
            }
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                "real": np.real(obj).tolist(),
                "imag": np.imag(obj).tolist(),
            }
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ============================================================
# Config override helpers
# ============================================================

def apply_common_overrides(
    cfg: ExperimentConfig,
    *,
    shots: Optional[int] = None,
    confusion_strength: Optional[float] = None,
    lambda_confusion: Optional[float] = None,
    beta: Optional[float] = None,
    gamma_rho: Optional[float] = None,
    gamma_c: Optional[float] = None,
    outer_max_iters: Optional[int] = None,
    inner_max_iters: Optional[int] = None,
    state_gd_max_iters: Optional[int] = None,
    confusion_gd_max_iters: Optional[int] = None,
    state_step_size: Optional[float] = None,
    confusion_step_size: Optional[float] = None,
    outer_tol: Optional[float] = None,
    inner_primal_tol: Optional[float] = None,
    inner_dual_tol: Optional[float] = None,
    state_gd_tol: Optional[float] = None,
    confusion_gd_tol: Optional[float] = None,
    use_shot_noise: Optional[bool] = None,
    loss_name: Optional[str] = None,
    prob_floor: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: Optional[bool] = None,
    print_every: Optional[int] = None,
    device: Optional[str] = None,
    real_dtype: Optional[str] = None,
    complex_dtype: Optional[str] = None,
    deterministic: Optional[bool] = None,
) -> ExperimentConfig:
    """
    Apply a common set of overrides to an experiment config in place.
    """
    if shots is not None:
        shots = _ensure_positive_int(shots, "shots")
        for region in cfg.regions:
            region.shots = shots

    if confusion_strength is not None:
        confusion_strength = _ensure_nonnegative_float(confusion_strength, "confusion_strength")
        for region in cfg.regions:
            region.confusion_strength = confusion_strength

    if lambda_confusion is not None:
        cfg.admm.lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")
    if beta is not None:
        cfg.admm.beta = _ensure_positive_float(beta, "beta")
    if gamma_rho is not None:
        cfg.admm.gamma_rho = _ensure_positive_float(gamma_rho, "gamma_rho")
    if gamma_c is not None:
        cfg.admm.gamma_c = _ensure_positive_float(gamma_c, "gamma_c")

    if outer_max_iters is not None:
        cfg.admm.outer_max_iters = _ensure_positive_int(outer_max_iters, "outer_max_iters")
    if inner_max_iters is not None:
        cfg.admm.inner_max_iters = _ensure_positive_int(inner_max_iters, "inner_max_iters")
    if state_gd_max_iters is not None:
        cfg.admm.state_gd_max_iters = _ensure_positive_int(state_gd_max_iters, "state_gd_max_iters")
    if confusion_gd_max_iters is not None:
        cfg.admm.confusion_gd_max_iters = _ensure_positive_int(confusion_gd_max_iters, "confusion_gd_max_iters")

    if state_step_size is not None:
        cfg.admm.state_step_size = _ensure_positive_float(state_step_size, "state_step_size")
    if confusion_step_size is not None:
        cfg.admm.confusion_step_size = _ensure_positive_float(confusion_step_size, "confusion_step_size")

    if outer_tol is not None:
        cfg.admm.outer_tol = _ensure_positive_float(outer_tol, "outer_tol")
    if inner_primal_tol is not None:
        cfg.admm.inner_primal_tol = _ensure_positive_float(inner_primal_tol, "inner_primal_tol")
    if inner_dual_tol is not None:
        cfg.admm.inner_dual_tol = _ensure_positive_float(inner_dual_tol, "inner_dual_tol")
    if state_gd_tol is not None:
        cfg.admm.state_gd_tol = _ensure_positive_float(state_gd_tol, "state_gd_tol")
    if confusion_gd_tol is not None:
        cfg.admm.confusion_gd_tol = _ensure_positive_float(confusion_gd_tol, "confusion_gd_tol")

    if use_shot_noise is not None:
        cfg.simulation.use_shot_noise = bool(use_shot_noise)

    if loss_name is not None:
        loss_name = _ensure_nonempty_string(loss_name, "loss_name")
        if loss_name not in SUPPORTED_LOSSES:
            raise ValueError(
                f"loss_name must be one of {sorted(SUPPORTED_LOSSES)}, got '{loss_name}'."
            )
        cfg.loss.name = loss_name

    if prob_floor is not None:
        cfg.loss.prob_floor = _ensure_positive_float(prob_floor, "prob_floor")

    if seed is not None:
        cfg.simulation.seed = int(seed)

    if verbose is not None:
        cfg.admm.verbose = bool(verbose)

    if print_every is not None:
        cfg.admm.print_every = _ensure_positive_int(print_every, "print_every")

    if device is not None:
        cfg.runtime.device = str(device)
    if real_dtype is not None:
        cfg.runtime.real_dtype = str(real_dtype)
    if complex_dtype is not None:
        cfg.runtime.complex_dtype = str(complex_dtype)
    if deterministic is not None:
        cfg.runtime.deterministic = bool(deterministic)

    return cfg


def apply_sweep_parameter(
    cfg: ExperimentConfig,
    parameter_name: str,
    parameter_value: Any,
) -> ExperimentConfig:
    """
    Apply one supported sweep parameter to the config in place.
    """
    parameter_name = _ensure_nonempty_string(parameter_name, "parameter_name")

    if parameter_name == "shots":
        return apply_common_overrides(cfg, shots=int(parameter_value))
    if parameter_name == "confusion_strength":
        return apply_common_overrides(cfg, confusion_strength=float(parameter_value))
    if parameter_name == "lambda_confusion":
        return apply_common_overrides(cfg, lambda_confusion=float(parameter_value))
    if parameter_name == "beta":
        return apply_common_overrides(cfg, beta=float(parameter_value))
    if parameter_name == "gamma_rho":
        return apply_common_overrides(cfg, gamma_rho=float(parameter_value))
    if parameter_name == "gamma_c":
        return apply_common_overrides(cfg, gamma_c=float(parameter_value))
    if parameter_name == "outer_max_iters":
        return apply_common_overrides(cfg, outer_max_iters=int(parameter_value))
    if parameter_name == "inner_max_iters":
        return apply_common_overrides(cfg, inner_max_iters=int(parameter_value))
    if parameter_name == "state_gd_max_iters":
        return apply_common_overrides(cfg, state_gd_max_iters=int(parameter_value))
    if parameter_name == "confusion_gd_max_iters":
        return apply_common_overrides(cfg, confusion_gd_max_iters=int(parameter_value))
    if parameter_name == "state_step_size":
        return apply_common_overrides(cfg, state_step_size=float(parameter_value))
    if parameter_name == "confusion_step_size":
        return apply_common_overrides(cfg, confusion_step_size=float(parameter_value))
    if parameter_name == "outer_tol":
        return apply_common_overrides(cfg, outer_tol=float(parameter_value))
    if parameter_name == "inner_primal_tol":
        return apply_common_overrides(cfg, inner_primal_tol=float(parameter_value))
    if parameter_name == "inner_dual_tol":
        return apply_common_overrides(cfg, inner_dual_tol=float(parameter_value))
    if parameter_name == "state_gd_tol":
        return apply_common_overrides(cfg, state_gd_tol=float(parameter_value))
    if parameter_name == "confusion_gd_tol":
        return apply_common_overrides(cfg, confusion_gd_tol=float(parameter_value))
    if parameter_name == "use_shot_noise":
        return apply_common_overrides(cfg, use_shot_noise=bool(parameter_value))
    if parameter_name == "loss_name":
        return apply_common_overrides(cfg, loss_name=str(parameter_value))
    if parameter_name == "prob_floor":
        return apply_common_overrides(cfg, prob_floor=float(parameter_value))
    if parameter_name == "seed":
        return apply_common_overrides(cfg, seed=int(parameter_value))
    if parameter_name == "device":
        return apply_common_overrides(cfg, device=str(parameter_value))
    if parameter_name == "real_dtype":
        return apply_common_overrides(cfg, real_dtype=str(parameter_value))
    if parameter_name == "complex_dtype":
        return apply_common_overrides(cfg, complex_dtype=str(parameter_value))
    if parameter_name == "deterministic":
        return apply_common_overrides(cfg, deterministic=bool(parameter_value))

    raise ValueError(f"Unsupported sweep parameter '{parameter_name}'.")


# ============================================================
# Metric extraction
# ============================================================

def extract_run_metrics(result: ExperimentRunResult) -> Dict[str, Any]:
    """
    Extract a flat dictionary of useful scalar metrics from a run result.
    """
    metrics: Dict[str, Any] = {}

    metrics["experiment_name"] = result.config.experiment_name
    metrics["converged"] = bool(result.solver_result.converged)
    metrics["outer_iterations"] = int(result.solver_result.num_outer_iterations)
    metrics["initial_objective"] = float(result.solver_result.initial_objective)
    metrics["final_objective"] = float(result.solver_result.final_objective)
    metrics["final_state_primal_residual"] = float(result.solver_result.final_state_primal_residual)
    metrics["final_state_dual_residual"] = float(result.solver_result.final_state_dual_residual)
    metrics["final_state_max_overlap_residual"] = float(result.solver_result.final_state_max_overlap_residual)
    metrics["final_confusion_average_pg_iters"] = float(result.solver_result.final_confusion_average_pg_iters)

    summary = result.summary

    metrics["fit_objective"] = _maybe_float(summary.get("fit_objective", math.nan))
    metrics["regularized_objective"] = _maybe_float(summary.get("regularized_objective", math.nan))

    overlap_summary = summary.get("overlap_consistency", {})
    metrics["overlap_mean"] = _maybe_float(overlap_summary.get("mean", math.nan))
    metrics["overlap_max"] = _maybe_float(overlap_summary.get("max", math.nan))
    metrics["overlap_min"] = _maybe_float(overlap_summary.get("min", math.nan))
    metrics["overlap_rms"] = _maybe_float(overlap_summary.get("rms", math.nan))

    state_error = summary.get("state_error", {})
    state_agg = state_error.get("aggregate", {}) if isinstance(state_error, dict) else {}
    metrics["state_error_mean"] = _maybe_float(state_agg.get("mean", math.nan))
    metrics["state_error_max"] = _maybe_float(state_agg.get("max", math.nan))
    metrics["state_error_min"] = _maybe_float(state_agg.get("min", math.nan))
    metrics["state_error_rms"] = _maybe_float(state_agg.get("rms", math.nan))

    confusion_error = summary.get("confusion_error", {})
    confusion_agg = confusion_error.get("aggregate", {}) if isinstance(confusion_error, dict) else {}
    metrics["confusion_error_mean"] = _maybe_float(confusion_agg.get("mean", math.nan))
    metrics["confusion_error_max"] = _maybe_float(confusion_agg.get("max", math.nan))
    metrics["confusion_error_min"] = _maybe_float(confusion_agg.get("min", math.nan))
    metrics["confusion_error_rms"] = _maybe_float(confusion_agg.get("rms", math.nan))

    prob_l2 = summary.get("probability_fit_l2", {})
    prob_l2_agg = prob_l2.get("aggregate", {}) if isinstance(prob_l2, dict) else {}
    metrics["prob_l2_mean"] = _maybe_float(prob_l2_agg.get("mean", math.nan))
    metrics["prob_l2_max"] = _maybe_float(prob_l2_agg.get("max", math.nan))
    metrics["prob_l2_min"] = _maybe_float(prob_l2_agg.get("min", math.nan))
    metrics["prob_l2_rms"] = _maybe_float(prob_l2_agg.get("rms", math.nan))

    prob_tv = summary.get("probability_fit_tv", {})
    prob_tv_agg = prob_tv.get("aggregate", {}) if isinstance(prob_tv, dict) else {}
    metrics["prob_tv_mean"] = _maybe_float(prob_tv_agg.get("mean", math.nan))
    metrics["prob_tv_max"] = _maybe_float(prob_tv_agg.get("max", math.nan))
    metrics["prob_tv_min"] = _maybe_float(prob_tv_agg.get("min", math.nan))
    metrics["prob_tv_rms"] = _maybe_float(prob_tv_agg.get("rms", math.nan))

    return metrics


def summarize_sweep_records(
    records: Sequence[Mapping[str, Any]],
    *,
    x_key: str = "parameter_value",
) -> Dict[Any, Dict[str, float]]:
    """
    Aggregate sweep records by the chosen x-axis key.
    """
    if len(records) == 0:
        raise ValueError("records must be non-empty.")

    grouped: Dict[Any, List[Mapping[str, Any]]] = {}
    for rec in records:
        x_val = rec[x_key]
        grouped.setdefault(x_val, []).append(rec)

    metric_keys = [
        "runtime_sec",
        "final_objective",
        "fit_objective",
        "regularized_objective",
        "state_error_mean",
        "state_error_max",
        "confusion_error_mean",
        "confusion_error_max",
        "prob_l2_mean",
        "prob_l2_max",
        "prob_tv_mean",
        "prob_tv_max",
        "overlap_mean",
        "overlap_max",
        "final_state_primal_residual",
        "final_state_dual_residual",
        "final_state_max_overlap_residual",
        "outer_iterations",
    ]

    out: Dict[Any, Dict[str, float]] = {}
    for x_val, rows in grouped.items():
        row_summary: Dict[str, float] = {}
        converged_vals = np.array([1.0 if bool(r["converged"]) else 0.0 for r in rows], dtype=float)
        row_summary["num_trials"] = float(len(rows))
        row_summary["convergence_rate"] = float(np.mean(converged_vals))

        for key in metric_keys:
            vals = np.array(
                [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))],
                dtype=float,
            )
            if vals.size == 0:
                row_summary[f"{key}_mean"] = math.nan
                row_summary[f"{key}_std"] = math.nan
                row_summary[f"{key}_min"] = math.nan
                row_summary[f"{key}_max"] = math.nan
            else:
                row_summary[f"{key}_mean"] = float(np.mean(vals))
                row_summary[f"{key}_std"] = float(np.std(vals, ddof=0))
                row_summary[f"{key}_min"] = float(np.min(vals))
                row_summary[f"{key}_max"] = float(np.max(vals))

        out[x_val] = row_summary

    return out


# ============================================================
# Plot helpers
# ============================================================

def _save_current_figure(save_dir: Optional[str], filename: str) -> Optional[str]:
    if save_dir is None:
        return None
    _mkdir_if_needed(save_dir)
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    return out_path


def plot_single_run_histories(
    result: ExperimentRunResult,
    *,
    save_dir: Optional[str] = None,
    prefix: str = "run",
    show: bool = True,
) -> None:
    """
    Plot the main scalar histories from a single run.
    """
    history = result.solver_result.history

    if "objective" in history and len(history["objective"]) > 0:
        plt.figure()
        plt.plot(history["objective"], marker="o")
        plt.xlabel("Outer iteration")
        plt.ylabel("Objective")
        plt.title("Outer objective history")
        plt.grid(True, alpha=0.3)
        _save_current_figure(save_dir, f"{prefix}_objective.png")
        if show:
            plt.show()
        else:
            plt.close()

    if "state_change" in history and len(history["state_change"]) > 0:
        plt.figure()
        plt.plot(history["state_change"], marker="o", label="state change")
        if "confusion_change" in history:
            plt.plot(history["confusion_change"], marker="o", label="confusion change")
        if "objective_relative_change" in history:
            plt.plot(history["objective_relative_change"], marker="o", label="objective relative change")
        plt.xlabel("Outer iteration")
        plt.ylabel("Relative change")
        plt.title("Outer relative changes")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save_current_figure(save_dir, f"{prefix}_relative_changes.png")
        if show:
            plt.show()
        else:
            plt.close()

    if "state_primal_residual" in history and len(history["state_primal_residual"]) > 0:
        plt.figure()
        plt.plot(history["state_primal_residual"], marker="o", label="primal")
        if "state_dual_residual" in history:
            plt.plot(history["state_dual_residual"], marker="o", label="dual")
        if "state_max_overlap_residual" in history:
            plt.plot(history["state_max_overlap_residual"], marker="o", label="max overlap")
        plt.xlabel("Outer iteration")
        plt.ylabel("Residual")
        plt.title("Inner ADMM residual summaries")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save_current_figure(save_dir, f"{prefix}_state_residuals.png")
        if show:
            plt.show()
        else:
            plt.close()

    if "confusion_average_pg_iters" in history and len(history["confusion_average_pg_iters"]) > 0:
        plt.figure()
        plt.plot(history["confusion_average_pg_iters"], marker="o")
        plt.xlabel("Outer iteration")
        plt.ylabel("Average PG iterations")
        plt.title("Confusion-update PG iterations")
        plt.grid(True, alpha=0.3)
        _save_current_figure(save_dir, f"{prefix}_confusion_pg_iters.png")
        if show:
            plt.show()
        else:
            plt.close()


def plot_sweep_metric(
    records: Sequence[Mapping[str, Any]],
    metric_key: str,
    *,
    x_key: str = "parameter_value",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    show_error_bars: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot one metric from a sweep.

    The plot aggregates repeated trials by mean and optional standard-deviation bars.
    """
    summary = summarize_sweep_records(records, x_key=x_key)
    x_vals = sorted(
        summary.keys(),
        key=lambda x: float(x) if isinstance(x, (int, float, np.number)) else str(x),
    )

    y_mean = np.array([summary[x][f"{metric_key}_mean"] for x in x_vals], dtype=float)
    y_std = np.array([summary[x][f"{metric_key}_std"] for x in x_vals], dtype=float)

    plt.figure()
    if show_error_bars and np.any(np.isfinite(y_std)):
        plt.errorbar(x_vals, y_mean, yerr=y_std, marker="o")
    else:
        plt.plot(x_vals, y_mean, marker="o")

    plt.xlabel(xlabel if xlabel is not None else x_key)
    plt.ylabel(ylabel if ylabel is not None else metric_key)
    plt.title(title if title is not None else f"{metric_key} vs {x_key}")
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        _mkdir_if_needed(os.path.dirname(save_path) if os.path.dirname(save_path) else None)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# Single run
# ============================================================

def run_single_experiment(
    config_or_name: Union[str, ExperimentConfig] = "fast_debug",
    *,
    truth_mode: str = "global_consistent",
    site_model_override: Optional[str] = None,
    shots: Optional[int] = None,
    confusion_strength: Optional[float] = None,
    lambda_confusion: Optional[float] = None,
    beta: Optional[float] = None,
    gamma_rho: Optional[float] = None,
    gamma_c: Optional[float] = None,
    outer_max_iters: Optional[int] = None,
    inner_max_iters: Optional[int] = None,
    use_shot_noise: Optional[bool] = None,
    loss_name: Optional[str] = None,
    prob_floor: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: Optional[bool] = None,
    make_plots: bool = True,
    save_dir: Optional[str] = None,
    save_json: bool = False,
) -> ExperimentRunResult:
    """
    Run one complete experiment and optionally make plots.

    This is the main one-call entry point for a single simulation setup.
    """
    cfg = _config_from_input(config_or_name)
    apply_common_overrides(
        cfg,
        shots=shots,
        confusion_strength=confusion_strength,
        lambda_confusion=lambda_confusion,
        beta=beta,
        gamma_rho=gamma_rho,
        gamma_c=gamma_c,
        outer_max_iters=outer_max_iters,
        inner_max_iters=inner_max_iters,
        use_shot_noise=use_shot_noise,
        loss_name=loss_name,
        prob_floor=prob_floor,
        seed=seed,
        verbose=verbose,
    )

    run_result = run_configured_experiment(
        cfg,
        truth_mode=truth_mode,
        site_model_override=site_model_override,
        loss=cfg.loss,
        prob_floor=cfg.loss.prob_floor,
        verbose=cfg.admm.verbose,
    )

    run_result.pretty_print()

    if make_plots:
        plot_single_run_histories(
            run_result,
            save_dir=save_dir,
            prefix=cfg.experiment_name,
            show=True,
        )

    if save_json:
        save_single_run_summary(
            run_result,
            save_dir=save_dir if save_dir is not None else ".",
            filename=f"{cfg.experiment_name}_summary.json",
        )

    return run_result


# ============================================================
# Parameter sweep
# ============================================================

def run_parameter_sweep(
    config_or_name: Union[str, ExperimentConfig],
    parameter_name: str,
    values: Sequence[Any],
    *,
    num_trials: int = 1,
    truth_mode: str = "global_consistent",
    site_model_override: Optional[str] = None,
    base_seed: Optional[int] = None,
    verbose: bool = False,
    **base_overrides: Any,
) -> List[Dict[str, Any]]:
    """
    Run a parameter sweep over one supported parameter.

    Returns
    -------
    list[dict]
        One record per run, including parameter value, trial index, runtime, and metrics.
    """
    parameter_name = _ensure_nonempty_string(parameter_name, "parameter_name")
    if len(values) == 0:
        raise ValueError("values must be a non-empty sequence.")
    num_trials = _ensure_positive_int(num_trials, "num_trials")

    base_cfg = _config_from_input(config_or_name)
    apply_common_overrides(base_cfg, **base_overrides)

    if base_seed is None:
        base_seed = int(base_cfg.simulation.seed)
    else:
        base_seed = int(base_seed)

    records: List[Dict[str, Any]] = []

    for value_idx, value in enumerate(values):
        for trial in range(num_trials):
            cfg = _deepcopy_config(base_cfg)

            run_seed = base_seed + 1000 * value_idx + trial
            cfg.simulation.seed = int(run_seed)

            apply_sweep_parameter(cfg, parameter_name, value)
            cfg.experiment_name = (
                f"{base_cfg.experiment_name}_{parameter_name}_{str(value).replace(' ', '_')}_trial_{trial}"
            )

            t0 = time.perf_counter()
            run_result = run_configured_experiment(
                cfg,
                truth_mode=truth_mode,
                site_model_override=site_model_override,
                loss=cfg.loss,
                prob_floor=cfg.loss.prob_floor,
                verbose=False,
            )
            runtime_sec = time.perf_counter() - t0

            record = extract_run_metrics(run_result)
            record["parameter_name"] = parameter_name
            record["parameter_value"] = value
            record["trial"] = trial
            record["seed"] = run_seed
            record["runtime_sec"] = float(runtime_sec)

            records.append(record)

            if verbose:
                print(
                    f"[sweep] {parameter_name}={value} "
                    f"trial={trial} "
                    f"runtime={runtime_sec:.3f}s "
                    f"final_obj={record['final_objective']:.6e} "
                    f"state_err={record.get('state_error_mean', math.nan):.6e}"
                )

    return records


# ============================================================
# Saving helpers
# ============================================================

def save_single_run_summary(
    result: ExperimentRunResult,
    *,
    save_dir: str,
    filename: str = "run_summary.json",
) -> str:
    """
    Save a single run summary, config, and history to JSON.

    Returns
    -------
    str
        Output path.
    """
    _mkdir_if_needed(save_dir)
    payload = {
        "config": result.config.to_dict(),
        "summary": result.summary,
        "history_summary": result.history_summary,
        "solver_history": result.solver_result.history,
        "solver_scalars": {
            "converged": result.solver_result.converged,
            "num_outer_iterations": result.solver_result.num_outer_iterations,
            "initial_objective": result.solver_result.initial_objective,
            "final_objective": result.solver_result.final_objective,
            "final_state_primal_residual": result.solver_result.final_state_primal_residual,
            "final_state_dual_residual": result.solver_result.final_state_dual_residual,
            "final_state_max_overlap_residual": result.solver_result.final_state_max_overlap_residual,
            "final_confusion_average_pg_iters": result.solver_result.final_confusion_average_pg_iters,
        },
    }
    out_path = os.path.join(save_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2)
    return out_path


def save_sweep_records_json(
    records: Sequence[Mapping[str, Any]],
    *,
    save_dir: str,
    filename: str = "sweep_records.json",
) -> str:
    """
    Save sweep records to JSON.

    Returns
    -------
    str
        Output path.
    """
    _mkdir_if_needed(save_dir)
    out_path = os.path.join(save_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(list(records)), f, indent=2)
    return out_path


def save_sweep_records_csv(
    records: Sequence[Mapping[str, Any]],
    *,
    save_dir: str,
    filename: str = "sweep_records.csv",
) -> str:
    """
    Save sweep records to CSV.

    Returns
    -------
    str
        Output path.
    """
    if len(records) == 0:
        raise ValueError("records must be non-empty.")

    _mkdir_if_needed(save_dir)
    out_path = os.path.join(save_dir, filename)

    fieldnames = sorted({str(k) for rec in records for k in rec.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})

    return out_path


# ============================================================
# Example entry points
# ============================================================

def demo_single_run() -> ExperimentRunResult:
    """
    Quick demo single run using the fast debug preset.
    """
    return run_single_experiment(
        "fast_debug",
        make_plots=True,
        verbose=False,
    )


def demo_sweep_shots(
    values: Sequence[int] = (200, 400, 800, 1600),
    *,
    num_trials: int = 2,
    save_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Quick demo sweep over the number of shots.
    """
    records = run_parameter_sweep(
        "fast_debug",
        parameter_name="shots",
        values=values,
        num_trials=num_trials,
        verbose=True,
    )

    plot_sweep_metric(
        records,
        metric_key="state_error_mean",
        x_key="parameter_value",
        xlabel="Shots",
        ylabel="Mean state error",
        title="Mean state error vs shots",
        save_path=None if save_dir is None else os.path.join(save_dir, "sweep_shots_state_error.png"),
        show=True,
    )

    if save_dir is not None:
        save_sweep_records_json(records, save_dir=save_dir, filename="sweep_shots.json")
        save_sweep_records_csv(records, save_dir=save_dir, filename="sweep_shots.csv")

    return records


def demo_sweep_noise(
    values: Sequence[float] = (0.0, 0.02, 0.05, 0.1),
    *,
    num_trials: int = 2,
    save_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Quick demo sweep over true readout-noise strength.
    """
    records = run_parameter_sweep(
        "fast_debug",
        parameter_name="confusion_strength",
        values=values,
        num_trials=num_trials,
        verbose=True,
    )

    plot_sweep_metric(
        records,
        metric_key="confusion_error_mean",
        x_key="parameter_value",
        xlabel="Confusion strength",
        ylabel="Mean confusion error",
        title="Mean confusion error vs readout-noise strength",
        save_path=None if save_dir is None else os.path.join(save_dir, "sweep_noise_confusion_error.png"),
        show=True,
    )

    if save_dir is not None:
        save_sweep_records_json(records, save_dir=save_dir, filename="sweep_noise.json")
        save_sweep_records_csv(records, save_dir=save_dir, filename="sweep_noise.csv")

    return records


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_common_overrides() -> None:
    cfg = make_named_experiment("fast_debug")
    apply_common_overrides(
        cfg,
        shots=321,
        confusion_strength=0.07,
        beta=3.0,
        gamma_rho=2.0,
        gamma_c=5.0,
        outer_max_iters=4,
        inner_max_iters=6,
        loss_name="l2",
        prob_floor=1e-9,
        seed=77,
        verbose=False,
    )

    assert cfg.loss.name == "l2"
    assert abs(cfg.loss.prob_floor - 1e-9) <= 1e-15
    assert cfg.simulation.seed == 77
    assert cfg.admm.beta == 3.0
    assert cfg.admm.gamma_rho == 2.0
    assert cfg.admm.gamma_c == 5.0
    assert cfg.admm.outer_max_iters == 4
    assert cfg.admm.inner_max_iters == 6
    assert all(region.shots == 321 for region in cfg.regions)
    assert all(abs(region.confusion_strength - 0.07) <= 1e-15 for region in cfg.regions)


def _self_test_extract_metrics() -> None:
    cfg = make_named_experiment("fast_debug")
    result = run_configured_experiment(cfg, truth_mode="global_consistent", verbose=False)
    metrics = extract_run_metrics(result)

    assert "final_objective" in metrics
    assert "fit_objective" in metrics
    assert "overlap_max" in metrics
    assert "state_error_mean" in metrics
    assert "confusion_error_mean" in metrics


def _self_test_sweep_summary() -> None:
    records = [
        {"parameter_value": 1, "converged": True, "runtime_sec": 1.0, "final_objective": 4.0, "fit_objective": 4.0,
         "regularized_objective": 4.0, "state_error_mean": 0.1, "state_error_max": 0.2,
         "confusion_error_mean": 0.3, "confusion_error_max": 0.4,
         "prob_l2_mean": 0.5, "prob_l2_max": 0.6, "prob_tv_mean": 0.7, "prob_tv_max": 0.8,
         "overlap_mean": 0.9, "overlap_max": 1.0, "final_state_primal_residual": 0.11,
         "final_state_dual_residual": 0.12, "final_state_max_overlap_residual": 0.13,
         "outer_iterations": 2},
        {"parameter_value": 1, "converged": False, "runtime_sec": 3.0, "final_objective": 6.0, "fit_objective": 6.0,
         "regularized_objective": 6.0, "state_error_mean": 0.3, "state_error_max": 0.4,
         "confusion_error_mean": 0.5, "confusion_error_max": 0.6,
         "prob_l2_mean": 0.7, "prob_l2_max": 0.8, "prob_tv_mean": 0.9, "prob_tv_max": 1.0,
         "overlap_mean": 1.1, "overlap_max": 1.2, "final_state_primal_residual": 0.21,
         "final_state_dual_residual": 0.22, "final_state_max_overlap_residual": 0.23,
         "outer_iterations": 3},
    ]
    summary = summarize_sweep_records(records)
    assert 1 in summary
    assert abs(summary[1]["num_trials"] - 2.0) <= 1e-12
    assert abs(summary[1]["convergence_rate"] - 0.5) <= 1e-12
    assert abs(summary[1]["runtime_sec_mean"] - 2.0) <= 1e-12


def run_self_tests(verbose: bool = True) -> None:
    tests = [
        ("common overrides", _self_test_common_overrides),
        ("metric extraction", _self_test_extract_metrics),
        ("sweep summary", _self_test_sweep_summary),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All main self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)

    print("\nAvailable experiments:")
    for name in list_available_experiments():
        print(f"  - {name}")

    print("\nRunning fast_debug demo...")
    demo = run_single_experiment(
        "fast_debug",
        make_plots=False,
        verbose=False,
        save_json=False,
    )
    demo.pretty_print()
