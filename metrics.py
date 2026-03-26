from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

from config import ExperimentConfig, LossConfig
from measurements import POVM, validate_region_povm_collection
from noise import validate_region_confusion_collection
from objectives import (
    build_region_shot_dict,
    region_prediction,
    total_data_fit_objective,
    total_regularized_objective,
)
from states import all_overlap_residuals, validate_region_state_collection


# ============================================================
# Small helpers
# ============================================================

def _as_torch_tensor(x, dtype=None, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(
            dtype=x.dtype if dtype is None else dtype,
            device=x.device if device is None else device,
        )
    return torch.as_tensor(x, dtype=dtype, device=device)


def _real_dtype_from_tensor(x: torch.Tensor) -> torch.dtype:
    if x.dtype in {torch.float32, torch.complex64}:
        return torch.float32
    if x.dtype in {torch.float64, torch.complex128}:
        return torch.float64
    return torch.float64


def _as_real_vector(x) -> torch.Tensor:
    t = _as_torch_tensor(x)
    if torch.is_complex(t):
        raise ValueError("Expected a real-valued vector.")
    if t.ndim == 0:
        t = t.reshape(1)
    return t.reshape(-1).to(dtype=_real_dtype_from_tensor(t))


def _ensure_nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _validate_same_region_keys(
    a: Mapping[str, object],
    b: Mapping[str, object],
    name_a: str,
    name_b: str,
) -> None:
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    if keys_a != keys_b:
        missing_from_b = keys_a - keys_b
        missing_from_a = keys_b - keys_a
        raise ValueError(
            f"{name_a} and {name_b} must have identical region keys. "
            f"Missing from {name_b}: {sorted(missing_from_b)}. "
            f"Missing from {name_a}: {sorted(missing_from_a)}."
        )


def _resolve_loss(
    loss: Optional[Union[str, LossConfig]],
    cfg: ExperimentConfig,
) -> Union[str, LossConfig]:
    return cfg.loss if loss is None else loss


def _resolve_prob_floor(
    loss: Union[str, LossConfig],
    prob_floor: Optional[float],
) -> Optional[float]:
    if prob_floor is not None:
        return float(prob_floor)
    if isinstance(loss, LossConfig):
        return float(loss.prob_floor)
    return None


# ============================================================
# Basic scalar / vector errors
# ============================================================

def frobenius_error(a, b) -> float:
    """
    Frobenius norm ||a - b||_F.
    """
    a_t = _as_torch_tensor(a)
    b_t = _as_torch_tensor(b, dtype=a_t.dtype, device=a_t.device)
    if tuple(a_t.shape) != tuple(b_t.shape):
        raise ValueError(f"Shapes must match, got {tuple(a_t.shape)} and {tuple(b_t.shape)}.")
    return float(torch.linalg.vector_norm((a_t - b_t).reshape(-1)).item())


def relative_frobenius_error(a, b, eps: float = 1e-12) -> float:
    """
    Relative Frobenius error ||a - b||_F / max(||b||_F, eps).
    """
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    a_t = _as_torch_tensor(a)
    b_t = _as_torch_tensor(b, dtype=a_t.dtype, device=a_t.device)
    if tuple(a_t.shape) != tuple(b_t.shape):
        raise ValueError(f"Shapes must match, got {tuple(a_t.shape)} and {tuple(b_t.shape)}.")

    num = torch.linalg.vector_norm((a_t - b_t).reshape(-1))
    den = torch.linalg.vector_norm(b_t.reshape(-1))
    return float((num / max(float(den.item()), eps)).item())


def _vector_l1(p, q) -> float:
    p_t = _as_real_vector(p)
    q_t = _as_real_vector(q).to(dtype=p_t.dtype, device=p_t.device)
    if tuple(p_t.shape) != tuple(q_t.shape):
        raise ValueError(f"Probability shapes must match, got {tuple(p_t.shape)} and {tuple(q_t.shape)}.")
    return float(torch.sum(torch.abs(p_t - q_t)).item())


def _vector_l2(p, q) -> float:
    p_t = _as_real_vector(p)
    q_t = _as_real_vector(q).to(dtype=p_t.dtype, device=p_t.device)
    if tuple(p_t.shape) != tuple(q_t.shape):
        raise ValueError(f"Probability shapes must match, got {tuple(p_t.shape)} and {tuple(q_t.shape)}.")
    return float(torch.linalg.vector_norm(p_t - q_t).item())


def l1_probability_error(p, q) -> float:
    """
    L1 probability error:
        ||p - q||_1.
    """
    return _vector_l1(p, q)


def l2_probability_error(p, q) -> float:
    """
    L2 probability error:
        ||p - q||_2.
    """
    return _vector_l2(p, q)


def total_variation_distance(p, q) -> float:
    """
    Total variation distance:
        TV(p, q) = 0.5 * ||p - q||_1.
    """
    return 0.5 * l1_probability_error(p, q)


def kl_divergence(p, q, floor: float = 1e-12) -> float:
    r"""
    KL divergence
        D_KL(p || q) = sum_i p_i log(p_i / q_i),
    with clipping floor for numerical stability.
    """
    floor = _ensure_nonnegative_float(floor, "floor")
    if floor <= 0.0:
        raise ValueError("floor must be strictly positive for KL divergence.")

    p_t = _as_real_vector(p)
    q_t = _as_real_vector(q).to(dtype=p_t.dtype, device=p_t.device)
    if tuple(p_t.shape) != tuple(q_t.shape):
        raise ValueError(f"Probability shapes must match, got {tuple(p_t.shape)} and {tuple(q_t.shape)}.")

    p_clipped = torch.clamp(p_t, min=float(floor))
    p_clipped = p_clipped / torch.sum(p_clipped)

    q_clipped = torch.clamp(q_t, min=float(floor))
    q_clipped = q_clipped / torch.sum(q_clipped)

    return float(torch.sum(p_clipped * torch.log(p_clipped / q_clipped)).item())


# ============================================================
# Region-collection error summaries
# ============================================================

def region_state_errors(
    estimated_states: Mapping[str, torch.Tensor],
    true_states: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Regionwise Frobenius errors between estimated and true states.
    """
    _validate_same_region_keys(estimated_states, true_states, "estimated_states", "true_states")
    return {
        name: frobenius_error(estimated_states[name], true_states[name])
        for name in estimated_states.keys()
    }


def region_state_relative_errors(
    estimated_states: Mapping[str, torch.Tensor],
    true_states: Mapping[str, torch.Tensor],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Regionwise relative Frobenius errors between estimated and true states.
    """
    _validate_same_region_keys(estimated_states, true_states, "estimated_states", "true_states")
    return {
        name: relative_frobenius_error(estimated_states[name], true_states[name], eps=eps)
        for name in estimated_states.keys()
    }


def aggregate_region_errors(error_dict: Mapping[str, float]) -> Dict[str, float]:
    """
    Aggregate a region-name -> scalar-error mapping.
    """
    values = torch.as_tensor(list(error_dict.values()), dtype=torch.float64)
    if values.numel() == 0:
        raise ValueError("error_dict must be non-empty.")
    return {
        "mean": float(torch.mean(values).item()),
        "max": float(torch.max(values).item()),
        "min": float(torch.min(values).item()),
        "median": float(torch.median(values).item()),
        "sum": float(torch.sum(values).item()),
        "rms": float(torch.sqrt(torch.mean(values ** 2)).item()),
    }


def region_confusion_errors(
    estimated_confusions: Mapping[str, torch.Tensor],
    true_confusions: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Regionwise Frobenius errors between estimated and true confusion matrices.
    """
    _validate_same_region_keys(
        estimated_confusions,
        true_confusions,
        "estimated_confusions",
        "true_confusions",
    )
    return {
        name: frobenius_error(estimated_confusions[name], true_confusions[name])
        for name in estimated_confusions.keys()
    }


def region_confusion_relative_errors(
    estimated_confusions: Mapping[str, torch.Tensor],
    true_confusions: Mapping[str, torch.Tensor],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Regionwise relative Frobenius errors between estimated and true confusion matrices.
    """
    _validate_same_region_keys(
        estimated_confusions,
        true_confusions,
        "estimated_confusions",
        "true_confusions",
    )
    return {
        name: relative_frobenius_error(
            estimated_confusions[name],
            true_confusions[name],
            eps=eps,
        )
        for name in estimated_confusions.keys()
    }


# ============================================================
# Probability-fit summaries
# ============================================================

def predicted_region_probabilities(
    region_states: Mapping[str, torch.Tensor],
    region_povms: Mapping[str, POVM],
    region_confusions: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Compute ideal and noisy predicted probabilities for each region.

    Returns
    -------
    tuple
        (ideal_probabilities, noisy_probabilities)
    """
    expected_names = set(region_states.keys())
    if set(region_povms.keys()) != expected_names:
        raise ValueError("region_povms keys must match region_states keys.")
    if set(region_confusions.keys()) != expected_names:
        raise ValueError("region_confusions keys must match region_states keys.")

    ideal: Dict[str, torch.Tensor] = {}
    noisy: Dict[str, torch.Tensor] = {}

    for name in region_states.keys():
        p_ideal, p_noisy = region_prediction(
            rho=region_states[name],
            povm=region_povms[name],
            confusion=region_confusions[name],
        )
        ideal[name] = p_ideal
        noisy[name] = p_noisy

    return ideal, noisy


def region_probability_fit_errors(
    empirical_probabilities: Mapping[str, torch.Tensor],
    predicted_probabilities: Mapping[str, torch.Tensor],
    metric: str = "l2",
    floor: float = 1e-12,
) -> Dict[str, float]:
    """
    Regionwise probability-fit errors between empirical and predicted probabilities.

    Supported metrics
    -----------------
    - "l1"
    - "l2"
    - "tv"
    - "kl"
    """
    metric = str(metric).lower()
    _validate_same_region_keys(
        empirical_probabilities,
        predicted_probabilities,
        "empirical_probabilities",
        "predicted_probabilities",
    )

    out: Dict[str, float] = {}
    for name in empirical_probabilities.keys():
        p = empirical_probabilities[name]
        q = predicted_probabilities[name]

        if metric == "l1":
            out[name] = l1_probability_error(p, q)
        elif metric == "l2":
            out[name] = l2_probability_error(p, q)
        elif metric == "tv":
            out[name] = total_variation_distance(p, q)
        elif metric == "kl":
            out[name] = kl_divergence(p, q, floor=floor)
        else:
            raise ValueError(f"Unsupported metric '{metric}'.")
    return out


# ============================================================
# Objective wrappers
# ============================================================

def evaluate_fit_objective(
    cfg: ExperimentConfig,
    empirical_probabilities: Mapping[str, torch.Tensor],
    region_states: Mapping[str, torch.Tensor],
    region_povms: Mapping[str, POVM],
    region_confusions: Mapping[str, torch.Tensor],
    *,
    loss: Optional[Union[str, LossConfig]] = None,
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
) -> float:
    """
    Wrapper for the total data-fit objective.
    """
    loss = _resolve_loss(loss, cfg)
    prob_floor = _resolve_prob_floor(loss, prob_floor)
    if region_shots is None:
        region_shots = build_region_shot_dict(cfg)

    return total_data_fit_objective(
        empirical_probabilities=empirical_probabilities,
        region_states=region_states,
        region_povms=region_povms,
        region_confusions=region_confusions,
        loss=loss,
        region_shots=region_shots,
        prob_floor=prob_floor,
    )


def evaluate_regularized_objective(
    cfg: ExperimentConfig,
    empirical_probabilities: Mapping[str, torch.Tensor],
    region_states: Mapping[str, torch.Tensor],
    region_povms: Mapping[str, POVM],
    region_confusions: Mapping[str, torch.Tensor],
    reference_confusions: Mapping[str, torch.Tensor],
    *,
    lambda_confusion: Optional[float] = None,
    loss: Optional[Union[str, LossConfig]] = None,
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
) -> float:
    """
    Wrapper for the total regularized objective.
    """
    loss = _resolve_loss(loss, cfg)
    prob_floor = _resolve_prob_floor(loss, prob_floor)
    if region_shots is None:
        region_shots = build_region_shot_dict(cfg)
    if lambda_confusion is None:
        lambda_confusion = cfg.admm.lambda_confusion

    return total_regularized_objective(
        empirical_probabilities=empirical_probabilities,
        region_states=region_states,
        region_povms=region_povms,
        region_confusions=region_confusions,
        reference_confusions=reference_confusions,
        lambda_confusion=float(lambda_confusion),
        loss=loss,
        region_shots=region_shots,
        prob_floor=prob_floor,
    )


# ============================================================
# Overlap summaries
# ============================================================

def overlap_consistency_summary(
    cfg: ExperimentConfig,
    region_states: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    """
    Summarize pairwise overlap residuals across all overlapping regions.
    """
    validate_region_state_collection(cfg, dict(region_states), check_overlap_consistency=False)
    per_pair = all_overlap_residuals(cfg, region_states)

    if len(per_pair) == 0:
        return {
            "num_overlaps": 0,
            "per_pair": {},
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "sum": 0.0,
            "rms": 0.0,
        }

    vals = torch.as_tensor(list(per_pair.values()), dtype=torch.float64)
    return {
        "num_overlaps": int(len(per_pair)),
        "per_pair": dict(per_pair),
        "mean": float(torch.mean(vals).item()),
        "max": float(torch.max(vals).item()),
        "min": float(torch.min(vals).item()),
        "sum": float(torch.sum(vals).item()),
        "rms": float(torch.sqrt(torch.mean(vals ** 2)).item()),
    }


# ============================================================
# History summaries
# ============================================================

def summarize_history(history: Mapping[str, Sequence[float]]) -> Dict[str, Dict[str, float]]:
    """
    Summarize scalar-history arrays.

    Returns
    -------
    dict
        Mapping history-key -> summary statistics.
    """
    out: Dict[str, Dict[str, float]] = {}

    for key, seq in history.items():
        arr = torch.as_tensor(list(seq), dtype=torch.float64).reshape(-1)
        if arr.numel() == 0:
            out[key] = {
                "num_points": 0.0,
                "initial": float("nan"),
                "final": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
            }
        else:
            out[key] = {
                "num_points": float(arr.numel()),
                "initial": float(arr[0].item()),
                "final": float(arr[-1].item()),
                "min": float(torch.min(arr).item()),
                "max": float(torch.max(arr).item()),
                "mean": float(torch.mean(arr).item()),
            }

    return out


# ============================================================
# End-to-end solution summary
# ============================================================

def summarize_solution(
    cfg: ExperimentConfig,
    empirical_probabilities: Mapping[str, torch.Tensor],
    region_states: Mapping[str, torch.Tensor],
    region_povms: Mapping[str, POVM],
    region_confusions: Mapping[str, torch.Tensor],
    *,
    reference_confusions: Optional[Mapping[str, torch.Tensor]] = None,
    true_region_states: Optional[Mapping[str, torch.Tensor]] = None,
    true_region_confusions: Optional[Mapping[str, torch.Tensor]] = None,
    loss: Optional[Union[str, LossConfig]] = None,
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
    lambda_confusion: Optional[float] = None,
) -> Dict[str, object]:
    """
    Build a compact metrics summary for an estimated solution.

    The summary includes:
    - objective values
    - overlap consistency
    - empirical-vs-predicted probability mismatches
    - optional truth-vs-estimate state/confusion errors
    """
    loss = _resolve_loss(loss, cfg)
    prob_floor = _resolve_prob_floor(loss, prob_floor)

    validate_region_state_collection(cfg, dict(region_states), check_overlap_consistency=False)
    validate_region_confusion_collection(cfg, dict(region_confusions))
    validate_region_povm_collection(cfg, dict(region_povms))

    if region_shots is None:
        region_shots = build_region_shot_dict(cfg)

    ideal_pred, noisy_pred = predicted_region_probabilities(
        region_states=region_states,
        region_povms=region_povms,
        region_confusions=region_confusions,
    )

    fit_value = evaluate_fit_objective(
        cfg=cfg,
        empirical_probabilities=empirical_probabilities,
        region_states=region_states,
        region_povms=region_povms,
        region_confusions=region_confusions,
        loss=loss,
        region_shots=region_shots,
        prob_floor=prob_floor,
    )

    summary: Dict[str, object] = {
        "fit_objective": float(fit_value),
        "overlap_consistency": overlap_consistency_summary(cfg, region_states),
        "probability_fit_l2": {
            "per_region": region_probability_fit_errors(
                empirical_probabilities,
                noisy_pred,
                metric="l2",
            ),
        },
        "probability_fit_tv": {
            "per_region": region_probability_fit_errors(
                empirical_probabilities,
                noisy_pred,
                metric="tv",
            ),
        },
    }

    summary["probability_fit_l2"]["aggregate"] = aggregate_region_errors(
        summary["probability_fit_l2"]["per_region"]
    )
    summary["probability_fit_tv"]["aggregate"] = aggregate_region_errors(
        summary["probability_fit_tv"]["per_region"]
    )

    if reference_confusions is not None:
        validate_region_confusion_collection(cfg, dict(reference_confusions))
        reg_value = evaluate_regularized_objective(
            cfg=cfg,
            empirical_probabilities=empirical_probabilities,
            region_states=region_states,
            region_povms=region_povms,
            region_confusions=region_confusions,
            reference_confusions=reference_confusions,
            lambda_confusion=lambda_confusion,
            loss=loss,
            region_shots=region_shots,
            prob_floor=prob_floor,
        )
        summary["regularized_objective"] = float(reg_value)

    if true_region_states is not None:
        validate_region_state_collection(cfg, dict(true_region_states), check_overlap_consistency=False)
        per_region_state = region_state_errors(region_states, true_region_states)
        summary["state_error"] = {
            "per_region": per_region_state,
            "aggregate": aggregate_region_errors(per_region_state),
        }

    if true_region_confusions is not None:
        validate_region_confusion_collection(cfg, dict(true_region_confusions))
        per_region_conf = region_confusion_errors(region_confusions, true_region_confusions)
        summary["confusion_error"] = {
            "per_region": per_region_conf,
            "aggregate": aggregate_region_errors(per_region_conf),
        }

    return summary


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_basic_probability_metrics() -> None:
    p = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    q = torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64)

    assert torch.isclose(torch.tensor(l1_probability_error(p, q)), torch.tensor(0.2))
    assert torch.isclose(torch.tensor(total_variation_distance(p, q)), torch.tensor(0.1))
    assert l2_probability_error(p, q) > 0.0
    assert kl_divergence(p, q) >= 0.0


def _self_test_region_error_aggregates() -> None:
    errs = {"A": 1.0, "B": 2.0, "C": 3.0}
    agg = aggregate_region_errors(errs)

    assert abs(agg["mean"] - 2.0) <= 1e-12
    assert abs(agg["max"] - 3.0) <= 1e-12
    assert abs(agg["min"] - 1.0) <= 1e-12
    assert abs(agg["sum"] - 6.0) <= 1e-12


def _self_test_overlap_summary() -> None:
    from config import make_default_experiment_config
    from states import generate_consistent_regional_truth_from_global_product

    cfg = make_default_experiment_config()
    _, _, region_states = generate_consistent_regional_truth_from_global_product(
        cfg=cfg,
        site_model="random_mixed",
        rng=123,
    )

    summary = overlap_consistency_summary(cfg, region_states)
    assert summary["num_overlaps"] == 1
    assert summary["max"] <= 1e-10


def _self_test_solution_summary() -> None:
    from config import make_default_experiment_config
    from noise import build_all_reference_confusions
    from simulator import simulate_experiment

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    sim = simulate_experiment(cfg, truth_mode="global_consistent")

    summary = summarize_solution(
        cfg=cfg,
        empirical_probabilities=sim.empirical_probabilities,
        region_states=sim.region_states,
        region_povms=sim.region_povms,
        region_confusions=sim.region_confusions,
        reference_confusions=build_all_reference_confusions(cfg),
        true_region_states=sim.region_states,
        true_region_confusions=sim.region_confusions,
        loss="l2",
        region_shots=sim.region_shots,
    )

    assert summary["fit_objective"] <= 1e-10
    assert summary["probability_fit_l2"]["aggregate"]["max"] <= 1e-10
    assert summary["state_error"]["aggregate"]["max"] <= 1e-10
    assert summary["confusion_error"]["aggregate"]["max"] <= 1e-10


def _self_test_history_summary() -> None:
    hist = {
        "objective": [5.0, 3.0, 2.0],
        "state_change": [1.0, 0.2, 0.05],
    }
    out = summarize_history(hist)

    assert abs(out["objective"]["initial"] - 5.0) <= 1e-12
    assert abs(out["objective"]["final"] - 2.0) <= 1e-12
    assert abs(out["state_change"]["max"] - 1.0) <= 1e-12


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the metrics module.
    """
    tests = [
        ("basic probability metrics", _self_test_basic_probability_metrics),
        ("region error aggregates", _self_test_region_error_aggregates),
        ("overlap summary", _self_test_overlap_summary),
        ("solution summary", _self_test_solution_summary),
        ("history summary", _self_test_history_summary),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All metrics self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
