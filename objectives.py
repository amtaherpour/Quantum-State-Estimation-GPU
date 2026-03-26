from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union

import torch

from config import ExperimentConfig, LossConfig
from core_ops import (
    DEFAULT_PROB_FLOOR,
    frobenius_norm,
    hermitian_part,
    hs_inner,
    partial_trace,
)
from measurements import POVM, measurement_map, measurement_map_adjoint
from noise import apply_confusion_matrix, confusion_frobenius_regularizer
from regions import RegionGraph


# ============================================================
# Small helpers
# ============================================================

def _real_dtype_for_tensor(x: torch.Tensor) -> torch.dtype:
    if x.dtype == torch.complex64:
        return torch.float32
    if x.dtype == torch.complex128:
        return torch.float64
    if x.dtype in {torch.float32, torch.float64}:
        return x.dtype
    raise ValueError(f"Unsupported dtype: {x.dtype}.")


def _coerce_device(device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _as_torch_tensor(x, dtype=None, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(
            dtype=x.dtype if dtype is None else dtype,
            device=x.device if device is None else device,
        )
    return torch.as_tensor(x, dtype=dtype, device=device)


def _as_real_vector(x, *, device=None, dtype=torch.float64) -> torch.Tensor:
    out = _as_torch_tensor(x, dtype=dtype, device=device).reshape(-1)
    if out.ndim != 1:
        raise ValueError("Expected a 1D vector.")
    return out


def _as_complex_matrix(x, *, device=None, dtype=torch.complex128) -> torch.Tensor:
    out = _as_torch_tensor(x, dtype=dtype, device=device)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {tuple(out.shape)}.")
    return out


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


def _resolve_loss_name(loss: Union[str, LossConfig]) -> str:
    return loss.name if isinstance(loss, LossConfig) else str(loss)


def _resolve_prob_floor(
    loss: Union[str, LossConfig],
    prob_floor: Optional[float],
) -> Optional[float]:
    if prob_floor is not None:
        return float(prob_floor)
    if isinstance(loss, LossConfig):
        return float(loss.prob_floor)
    return None


def _canonical_eta_key(
    graph: RegionGraph,
    region_a: Union[int, str],
    region_b: Union[int, str],
) -> Tuple[str, str]:
    ia = graph.region_index(region_a)
    ib = graph.region_index(region_b)
    if ia == ib:
        raise ValueError("Cannot build an eta key from identical regions.")
    if ia < ib:
        return (graph.region_name(ia), graph.region_name(ib))
    return (graph.region_name(ib), graph.region_name(ia))


def _directed_dual_key(
    graph: RegionGraph,
    region_a: Union[int, str],
    region_b: Union[int, str],
) -> Tuple[str, str]:
    ia = graph.region_index(region_a)
    ib = graph.region_index(region_b)
    if ia == ib:
        raise ValueError("Cannot build a dual key from identical regions.")
    return (graph.region_name(ia), graph.region_name(ib))


# ============================================================
# Discrepancy values and gradients
# ============================================================

def discrepancy_value(
    empirical,
    predicted,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
) -> float:
    """
    Discrepancy value D(empirical, predicted).

    Supported losses
    ----------------
    l2 :
        0.5 * ||predicted - empirical||_2^2
    nll :
        Multinomial negative log-likelihood up to an additive constant:
            -shots * empirical^T log(predicted)
        when `shots` is provided, or
            -empirical^T log(predicted)
        otherwise.
    """
    predicted_t = _as_torch_tensor(predicted)
    real_dtype = _real_dtype_for_tensor(predicted_t) if predicted_t.ndim > 0 else torch.float64
    device = predicted_t.device if isinstance(predicted, torch.Tensor) else None

    empirical_t = _as_real_vector(empirical, dtype=real_dtype, device=device)
    predicted_t = _as_real_vector(predicted, dtype=real_dtype, device=device)

    if empirical_t.shape != predicted_t.shape:
        raise ValueError(
            f"empirical and predicted must have the same shape, got "
            f"{tuple(empirical_t.shape)} and {tuple(predicted_t.shape)}."
        )

    loss_name = _resolve_loss_name(loss)

    if loss_name == "l2":
        diff = predicted_t - empirical_t
        return float(0.5 * torch.dot(diff, diff).item())

    if loss_name == "nll":
        floor = DEFAULT_PROB_FLOOR if prob_floor is None else _ensure_positive_float(prob_floor, "prob_floor")
        predicted_safe = torch.clamp(predicted_t, min=float(floor))
        weight = 1.0 if shots is None else float(shots)
        return float((-weight * torch.dot(empirical_t, torch.log(predicted_safe))).item())

    raise ValueError(f"Unsupported loss '{loss_name}'.")


def gradient_wrt_predicted(
    empirical,
    predicted,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
) -> torch.Tensor:
    """
    Gradient of the discrepancy with respect to the predicted probability vector.
    """
    predicted_t = _as_torch_tensor(predicted)
    real_dtype = _real_dtype_for_tensor(predicted_t) if predicted_t.ndim > 0 else torch.float64
    device = predicted_t.device if isinstance(predicted, torch.Tensor) else None

    empirical_t = _as_real_vector(empirical, dtype=real_dtype, device=device)
    predicted_t = _as_real_vector(predicted, dtype=real_dtype, device=device)

    if empirical_t.shape != predicted_t.shape:
        raise ValueError(
            f"empirical and predicted must have the same shape, got "
            f"{tuple(empirical_t.shape)} and {tuple(predicted_t.shape)}."
        )

    loss_name = _resolve_loss_name(loss)

    if loss_name == "l2":
        return predicted_t - empirical_t

    if loss_name == "nll":
        floor = DEFAULT_PROB_FLOOR if prob_floor is None else _ensure_positive_float(prob_floor, "prob_floor")
        predicted_safe = torch.clamp(predicted_t, min=float(floor))
        weight = 1.0 if shots is None else float(shots)
        return -weight * empirical_t / predicted_safe

    raise ValueError(f"Unsupported loss '{loss_name}'.")


# ============================================================
# Region-level prediction and gradients
# ============================================================

def region_prediction(
    rho,
    povm: POVM,
    confusion,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ideal and noisy regional prediction vectors:
        p_ideal = M(rho)
        p_noisy = C p_ideal
    """
    rho_t = _as_complex_matrix(rho, dtype=povm.dtype, device=povm.device)
    confusion_t = _as_torch_tensor(
        confusion,
        dtype=_real_dtype_for_tensor(povm.effects[0]),
        device=povm.device,
    )

    ideal = measurement_map(rho_t, povm)
    noisy = apply_confusion_matrix(confusion_t, ideal, prob_floor=0.0)
    return ideal, noisy


def region_fit_objective(
    empirical,
    rho,
    povm: POVM,
    confusion,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
) -> float:
    """
    Region-local data-fit objective:
        D(empirical, C M(rho)).
    """
    _, noisy = region_prediction(rho, povm, confusion)
    return discrepancy_value(
        empirical=empirical,
        predicted=noisy,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )


def region_gradient_components(
    empirical,
    rho,
    povm: POVM,
    confusion,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Return the main gradient components for one region.

    Returns a dictionary containing:
    - ideal_probabilities
    - noisy_probabilities
    - grad_predicted : dD/dq
    - grad_ideal     : dD/dp where q = C p
    - grad_rho       : dD/drho via M^*(grad_ideal)
    - grad_C         : dD/dC = grad_predicted @ ideal_probabilities^T
    """
    rho_t = _as_complex_matrix(rho, dtype=povm.dtype, device=povm.device)
    confusion_t = _as_torch_tensor(
        confusion,
        dtype=_real_dtype_for_tensor(povm.effects[0]),
        device=povm.device,
    )

    ideal, noisy = region_prediction(rho_t, povm, confusion_t)
    grad_predicted = gradient_wrt_predicted(
        empirical=empirical,
        predicted=noisy,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )
    grad_ideal = confusion_t.transpose(0, 1) @ grad_predicted
    grad_rho = hermitian_part(measurement_map_adjoint(grad_ideal, povm))
    grad_C = torch.outer(grad_predicted, ideal)

    return {
        "ideal_probabilities": ideal,
        "noisy_probabilities": noisy,
        "grad_predicted": grad_predicted,
        "grad_ideal": grad_ideal,
        "grad_rho": grad_rho,
        "grad_C": grad_C,
    }


# ============================================================
# Region-local proximal objectives
# ============================================================

def state_subproblem_region_objective(
    empirical,
    rho,
    povm: POVM,
    confusion,
    rho_prev,
    gamma_rho: float,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
) -> float:
    """
    Region-local state-update objective:
        D(empirical, C M(rho))
        + (gamma_rho / 2) ||rho - rho_prev||_F^2
    """
    gamma_rho = _ensure_positive_float(gamma_rho, "gamma_rho")

    rho_t = _as_complex_matrix(rho, dtype=povm.dtype, device=povm.device)
    rho_prev_t = _as_complex_matrix(rho_prev, dtype=povm.dtype, device=povm.device)

    fit = region_fit_objective(
        empirical=empirical,
        rho=rho_t,
        povm=povm,
        confusion=confusion,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )
    prox = 0.5 * gamma_rho * frobenius_norm(rho_t - rho_prev_t) ** 2
    return float(fit + prox)


def state_subproblem_region_gradient(
    empirical,
    rho,
    povm: POVM,
    confusion,
    rho_prev,
    gamma_rho: float,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
) -> torch.Tensor:
    """
    Gradient of the proximal regional state-update objective.
    """
    gamma_rho = _ensure_positive_float(gamma_rho, "gamma_rho")

    rho_t = _as_complex_matrix(rho, dtype=povm.dtype, device=povm.device)
    rho_prev_t = _as_complex_matrix(rho_prev, dtype=povm.dtype, device=povm.device)

    comps = region_gradient_components(
        empirical=empirical,
        rho=rho_t,
        povm=povm,
        confusion=confusion,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )
    return hermitian_part(comps["grad_rho"] + gamma_rho * (rho_t - rho_prev_t))


def confusion_subproblem_region_objective(
    empirical,
    ideal_probabilities,
    confusion,
    reference_confusion,
    lambda_confusion: float,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
    gamma_c: float = 0.0,
    confusion_prev=None,
) -> float:
    r"""
    Region-local confusion-update objective:
        D(empirical, C p_ideal)
        + lambda ||C - C_ref||_F^2
        + (gamma_c / 2) ||C - C_prev||_F^2
    """
    lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")
    gamma_c = _ensure_nonnegative_float(gamma_c, "gamma_c")

    confusion_t = _as_torch_tensor(confusion)
    real_dtype = _real_dtype_for_tensor(confusion_t)
    device = confusion_t.device

    ideal_t = _as_real_vector(ideal_probabilities, dtype=real_dtype, device=device)
    confusion_t = _as_torch_tensor(confusion, dtype=real_dtype, device=device)
    reference_t = _as_torch_tensor(reference_confusion, dtype=real_dtype, device=device)

    noisy = apply_confusion_matrix(confusion_t, ideal_t, prob_floor=0.0)

    val = discrepancy_value(
        empirical=empirical,
        predicted=noisy,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )
    val += lambda_confusion * confusion_frobenius_regularizer(confusion_t, reference_t)

    if gamma_c > 0.0:
        if confusion_prev is None:
            raise ValueError("confusion_prev must be provided when gamma_c > 0.")
        confusion_prev_t = _as_torch_tensor(confusion_prev, dtype=real_dtype, device=device)
        val += 0.5 * gamma_c * frobenius_norm(confusion_t - confusion_prev_t) ** 2

    return float(val)


# ============================================================
# Aggregation across regions
# ============================================================

def build_region_shot_dict(cfg: ExperimentConfig) -> Dict[str, int]:
    """
    Build a region-name -> shots dictionary from the config.
    """
    return {region.name: int(region.shots) for region in cfg.regions}


def total_data_fit_objective(
    empirical_probabilities: Mapping[str, torch.Tensor],
    region_states: Mapping[str, torch.Tensor],
    region_povms: Mapping[str, POVM],
    region_confusions: Mapping[str, torch.Tensor],
    loss: Union[str, LossConfig],
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
) -> float:
    """
    Sum of region-level data-fit objectives over all regions.
    """
    names = set(empirical_probabilities.keys())
    if set(region_states.keys()) != names:
        raise ValueError("region_states keys must match empirical_probabilities keys.")
    if set(region_povms.keys()) != names:
        raise ValueError("region_povms keys must match empirical_probabilities keys.")
    if set(region_confusions.keys()) != names:
        raise ValueError("region_confusions keys must match empirical_probabilities keys.")
    if region_shots is not None and set(region_shots.keys()) != names:
        raise ValueError("region_shots keys must match empirical_probabilities keys.")

    total = 0.0
    for name in names:
        shots = None if region_shots is None else int(region_shots[name])
        total += region_fit_objective(
            empirical=empirical_probabilities[name],
            rho=region_states[name],
            povm=region_povms[name],
            confusion=region_confusions[name],
            loss=loss,
            shots=shots,
            prob_floor=prob_floor,
        )
    return float(total)


def total_regularized_objective(
    empirical_probabilities: Mapping[str, torch.Tensor],
    region_states: Mapping[str, torch.Tensor],
    region_povms: Mapping[str, POVM],
    region_confusions: Mapping[str, torch.Tensor],
    reference_confusions: Mapping[str, torch.Tensor],
    lambda_confusion: float,
    loss: Union[str, LossConfig],
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
) -> float:
    r"""
    Total regularized objective:
        sum_r D(empirical_r, C_r M_r(rho_r))
        + lambda sum_r ||C_r - C_ref,r||_F^2
    """
    lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")

    names = set(empirical_probabilities.keys())
    if set(reference_confusions.keys()) != names:
        raise ValueError("reference_confusions keys must match empirical_probabilities keys.")

    total = total_data_fit_objective(
        empirical_probabilities=empirical_probabilities,
        region_states=region_states,
        region_povms=region_povms,
        region_confusions=region_confusions,
        loss=loss,
        region_shots=region_shots,
        prob_floor=prob_floor,
    )

    for name in names:
        total += lambda_confusion * confusion_frobenius_regularizer(
            region_confusions[name],
            reference_confusions[name],
        )

    return float(total)


# ============================================================
# ADMM overlap residual helpers
# ============================================================

def overlap_primal_residual_norm(
    graph: RegionGraph,
    region_states: Mapping[str, torch.Tensor],
    eta: Mapping[Tuple[str, str], torch.Tensor],
) -> float:
    """
    Global ADMM primal residual for overlap-consensus constraints.
    """
    total_sq = 0.0

    for i, j in graph.overlap_pairs():
        name_i = graph.region_name(i)
        name_j = graph.region_name(j)
        key = (name_i, name_j)

        info = graph.overlap_info(i, j)
        rho_i = region_states[name_i]
        rho_j = region_states[name_j]
        eta_ij = eta[key]

        red_i = partial_trace(rho_i, dims=graph.region_site_dims(i), keep=info.local_keep_i)
        red_j = partial_trace(rho_j, dims=graph.region_site_dims(j), keep=info.local_keep_j)

        total_sq += frobenius_norm(red_i - eta_ij) ** 2
        total_sq += frobenius_norm(red_j - eta_ij) ** 2

    return float(total_sq ** 0.5)


def overlap_dual_residual_norm(
    beta: float,
    eta_current: Mapping[Tuple[str, str], torch.Tensor],
    eta_previous: Mapping[Tuple[str, str], torch.Tensor],
) -> float:
    """
    Global ADMM dual residual:
        beta * ||eta_current - eta_previous||_F
    aggregated across all overlap variables.
    """
    beta = _ensure_positive_float(beta, "beta")

    if set(eta_current.keys()) != set(eta_previous.keys()):
        raise ValueError("eta_current and eta_previous must have identical keys.")

    total_sq = 0.0
    for key in eta_current.keys():
        total_sq += frobenius_norm(eta_current[key] - eta_previous[key]) ** 2

    return float(beta * (total_sq ** 0.5))


def max_overlap_residual(
    graph: RegionGraph,
    region_states: Mapping[str, torch.Tensor],
    eta: Mapping[Tuple[str, str], torch.Tensor],
) -> float:
    """
    Maximum Frobenius overlap-consensus residual across all individual constraints.
    """
    max_val = 0.0

    for i, j in graph.overlap_pairs():
        name_i = graph.region_name(i)
        name_j = graph.region_name(j)
        key = (name_i, name_j)
        info = graph.overlap_info(i, j)

        rho_i = region_states[name_i]
        rho_j = region_states[name_j]
        eta_ij = eta[key]

        red_i = partial_trace(rho_i, dims=graph.region_site_dims(i), keep=info.local_keep_i)
        red_j = partial_trace(rho_j, dims=graph.region_site_dims(j), keep=info.local_keep_j)

        max_val = max(max_val, frobenius_norm(red_i - eta_ij))
        max_val = max(max_val, frobenius_norm(red_j - eta_ij))

    return float(max_val)


# ============================================================
# Region-local augmented objective and gradient for ADMM
# ============================================================

def region_augmented_state_objective(
    graph: RegionGraph,
    region_name: str,
    rho,
    rho_outer_prev,
    empirical_probabilities: Mapping[str, torch.Tensor],
    region_povms: Mapping[str, POVM],
    fixed_confusions: Mapping[str, torch.Tensor],
    eta: Mapping[Tuple[str, str], torch.Tensor],
    duals: Mapping[Tuple[str, str], torch.Tensor],
    beta: float,
    gamma_rho: float,
    loss: Union[str, LossConfig],
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
) -> float:
    """
    Region-local objective used in the ADMM state update.
    """
    beta = _ensure_positive_float(beta, "beta")
    gamma_rho = _ensure_positive_float(gamma_rho, "gamma_rho")

    povm = region_povms[region_name]
    rho_t = _as_complex_matrix(rho, dtype=povm.dtype, device=povm.device)

    shots = None if region_shots is None else int(region_shots[region_name])

    obj = state_subproblem_region_objective(
        empirical=empirical_probabilities[region_name],
        rho=rho_t,
        povm=povm,
        confusion=fixed_confusions[region_name],
        rho_prev=rho_outer_prev,
        gamma_rho=gamma_rho,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )

    region_idx = graph.region_index(region_name)
    dims = graph.region_site_dims(region_idx)

    for neigh_idx in graph.neighbors(region_idx):
        eta_key = _canonical_eta_key(graph, region_idx, neigh_idx)
        dual_key = _directed_dual_key(graph, region_idx, neigh_idx)
        keep_self, _ = graph.local_keep_indices(region_idx, neigh_idx)

        red = partial_trace(rho_t, dims=dims, keep=keep_self)
        resid = red - eta[eta_key]

        obj += hs_inner(duals[dual_key], resid)
        obj += 0.5 * beta * frobenius_norm(resid) ** 2

    return float(obj)


def region_augmented_state_gradient(
    graph: RegionGraph,
    region_name: str,
    rho,
    rho_outer_prev,
    empirical_probabilities: Mapping[str, torch.Tensor],
    region_povms: Mapping[str, POVM],
    fixed_confusions: Mapping[str, torch.Tensor],
    eta: Mapping[Tuple[str, str], torch.Tensor],
    duals: Mapping[Tuple[str, str], torch.Tensor],
    beta: float,
    gamma_rho: float,
    loss: Union[str, LossConfig],
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
) -> torch.Tensor:
    """
    Gradient of the ADMM region-local augmented state objective.

    This uses torch autograd for the overlap-constraint terms so the full
    objective remains correct on GPU without hand-coding the adjoint of every
    subsystem reduction.
    """
    beta = _ensure_positive_float(beta, "beta")
    gamma_rho = _ensure_positive_float(gamma_rho, "gamma_rho")

    povm = region_povms[region_name]
    rho_init = _as_complex_matrix(rho, dtype=povm.dtype, device=povm.device)
    rho_var = rho_init.clone().detach().requires_grad_(True)

    shots = None if region_shots is None else int(region_shots[region_name])

    region_idx = graph.region_index(region_name)
    dims = graph.region_site_dims(region_idx)

    # Data-fit + proximal term
    ideal = measurement_map(rho_var, povm)
    noisy = apply_confusion_matrix(fixed_confusions[region_name], ideal, prob_floor=0.0)

    loss_value = discrepancy_value(
        empirical=empirical_probabilities[region_name],
        predicted=noisy,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )
    obj = torch.tensor(
        loss_value,
        dtype=_real_dtype_for_tensor(rho_var),
        device=rho_var.device,
    )
    obj = obj + 0.5 * gamma_rho * torch.sum(torch.abs(rho_var - _as_complex_matrix(
        rho_outer_prev,
        dtype=povm.dtype,
        device=povm.device,
    )) ** 2)

    # ADMM overlap terms
    for neigh_idx in graph.neighbors(region_idx):
        eta_key = _canonical_eta_key(graph, region_idx, neigh_idx)
        dual_key = _directed_dual_key(graph, region_idx, neigh_idx)
        keep_self, _ = graph.local_keep_indices(region_idx, neigh_idx)

        red = partial_trace(rho_var, dims=dims, keep=keep_self)
        resid = red - eta[eta_key]

        obj = obj + torch.real(torch.trace(duals[dual_key].conj().transpose(-2, -1) @ resid))
        obj = obj + 0.5 * beta * torch.sum(torch.abs(resid) ** 2)

    grad = torch.autograd.grad(obj, rho_var, create_graph=False, retain_graph=False)[0]
    return hermitian_part(grad)


# ============================================================
# Relative-change helpers
# ============================================================

def relative_change_dict(
    current: Mapping[str, torch.Tensor],
    previous: Mapping[str, torch.Tensor],
) -> float:
    """
    Aggregate relative change for a dictionary of matrices:
        ||x_k - x_{k-1}|| / max(||x_{k-1}||, eps)
    using a global Frobenius aggregation.
    """
    if set(current.keys()) != set(previous.keys()):
        raise ValueError("current and previous must have identical keys.")

    num_sq = 0.0
    den_sq = 0.0
    for key in current.keys():
        num_sq += frobenius_norm(current[key] - previous[key]) ** 2
        den_sq += frobenius_norm(previous[key]) ** 2

    denom = max(den_sq ** 0.5, 1e-12)
    return float((num_sq ** 0.5) / denom)


# ============================================================
# Lightweight self-tests
# ============================================================

def _finite_difference_check_grad_predicted_l2() -> None:
    empirical = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    predicted = torch.tensor([0.3, 0.4, 0.3], dtype=torch.float64)

    grad = gradient_wrt_predicted(empirical, predicted, loss="l2")
    eps = 1e-7

    direction = torch.tensor([0.4, -0.2, 0.1], dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)

    f_plus = discrepancy_value(empirical, predicted + eps * direction, loss="l2")
    f_minus = discrepancy_value(empirical, predicted - eps * direction, loss="l2")
    fd = (f_plus - f_minus) / (2.0 * eps)
    ip = float(torch.dot(grad, direction).item())

    assert abs(fd - ip) <= 1e-6


def _finite_difference_check_grad_predicted_nll() -> None:
    empirical = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    predicted = torch.tensor([0.31, 0.39, 0.30], dtype=torch.float64)

    grad = gradient_wrt_predicted(empirical, predicted, loss="nll", prob_floor=1e-12)
    eps = 1e-7

    direction = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)

    f_plus = discrepancy_value(empirical, predicted + eps * direction, loss="nll", prob_floor=1e-12)
    f_minus = discrepancy_value(empirical, predicted - eps * direction, loss="nll", prob_floor=1e-12)
    fd = (f_plus - f_minus) / (2.0 * eps)
    ip = float(torch.dot(grad, direction).item())

    assert abs(fd - ip) <= 1e-5


def _self_test_region_gradient_shapes() -> None:
    from measurements import make_computational_povm
    from noise import make_noisy_identity_confusion

    rho = torch.tensor([[0.7, 0.0], [0.0, 0.3]], dtype=torch.complex128)
    povm = make_computational_povm(2)
    c = make_noisy_identity_confusion(2, strength=0.1)
    empirical = torch.tensor([0.6, 0.4], dtype=torch.float64)

    out = region_gradient_components(
        empirical=empirical,
        rho=rho,
        povm=povm,
        confusion=c,
        loss="l2",
    )

    assert out["ideal_probabilities"].shape == (2,)
    assert out["noisy_probabilities"].shape == (2,)
    assert out["grad_predicted"].shape == (2,)
    assert out["grad_ideal"].shape == (2,)
    assert out["grad_rho"].shape == (2, 2)
    assert out["grad_C"].shape == (2, 2)


def _self_test_total_objective_l2_zero_fit() -> None:
    from config import make_default_experiment_config
    from measurements import build_all_region_povms
    from noise import build_all_reference_confusions, build_all_true_confusions
    from states import generate_consistent_regional_truth_from_global_product

    cfg = make_default_experiment_config()
    _, _, region_states = generate_consistent_regional_truth_from_global_product(
        cfg=cfg,
        site_model="random_mixed",
        rng=123,
    )
    povms = build_all_region_povms(cfg, rng=123)
    confusions = build_all_true_confusions(cfg, rng=123)
    refs = build_all_reference_confusions(cfg)

    empirical = {}
    for name in region_states.keys():
        _, noisy = region_prediction(region_states[name], povms[name], confusions[name])
        empirical[name] = noisy

    val_fit = total_data_fit_objective(
        empirical_probabilities=empirical,
        region_states=region_states,
        region_povms=povms,
        region_confusions=confusions,
        loss="l2",
    )
    assert abs(val_fit) <= 1e-10

    val_reg = total_regularized_objective(
        empirical_probabilities=empirical,
        region_states=region_states,
        region_povms=povms,
        region_confusions=confusions,
        reference_confusions=refs,
        lambda_confusion=0.0,
        loss="l2",
    )
    assert abs(val_reg) <= 1e-10


def _self_test_overlap_residual_helpers() -> None:
    from config import make_default_experiment_config
    from states import generate_consistent_regional_truth_from_global_product

    cfg = make_default_experiment_config()
    graph = RegionGraph(cfg)
    _, _, region_states = generate_consistent_regional_truth_from_global_product(
        cfg=cfg,
        site_model="random_mixed",
        rng=321,
    )

    eta = {}
    for i, j in graph.overlap_pairs():
        info = graph.overlap_info(i, j)
        name_i = graph.region_name(i)
        rho_i = region_states[name_i]
        eta[(name_i, graph.region_name(j))] = partial_trace(
            rho_i,
            dims=graph.region_site_dims(i),
            keep=info.local_keep_i,
        )

    primal = overlap_primal_residual_norm(graph, region_states, eta)
    dual = overlap_dual_residual_norm(beta=1.0, eta_current=eta, eta_previous=eta)

    assert abs(primal) <= 1e-10
    assert abs(dual) <= 1e-10


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the objectives module.
    """
    tests = [
        ("L2 gradient wrt predicted", _finite_difference_check_grad_predicted_l2),
        ("NLL gradient wrt predicted", _finite_difference_check_grad_predicted_nll),
        ("region gradient shapes", _self_test_region_gradient_shapes),
        ("total objective zero-fit check", _self_test_total_objective_l2_zero_fit),
        ("overlap residual helpers", _self_test_overlap_residual_helpers),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All objectives self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
