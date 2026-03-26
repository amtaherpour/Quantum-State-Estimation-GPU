from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from config import ExperimentConfig, LossConfig, RegionConfig
from core_ops import (
    frobenius_norm,
    hermitian_part,
    hs_inner,
    identity,
    is_density_matrix,
    partial_trace,
    permute_subsystems,
    project_to_density_matrix,
)
from measurements import POVM
from objectives import (
    build_region_shot_dict,
    max_overlap_residual,
    overlap_dual_residual_norm,
    overlap_primal_residual_norm,
    region_gradient_components,
    state_subproblem_region_objective,
)
from regions import RegionGraph
from states import validate_region_state_collection


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


def _real_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    if dtype in {torch.float32, torch.float64}:
        return dtype
    raise ValueError(f"Unsupported dtype: {dtype}.")


def _complex_dtype_for_tensor(x: torch.Tensor) -> torch.dtype:
    if x.dtype == torch.complex64:
        return torch.complex64
    if x.dtype == torch.complex128:
        return torch.complex128
    if x.dtype == torch.float32:
        return torch.complex64
    if x.dtype == torch.float64:
        return torch.complex128
    return torch.complex128


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


def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _resolve_loss(loss: Optional[Union[str, LossConfig]], cfg: ExperimentConfig) -> Union[str, LossConfig]:
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


def _copy_matrix_dict(d: Mapping) -> Dict:
    return {k: _as_torch_tensor(v).clone() for k, v in d.items()}


def _directed_dual_key(
    graph: RegionGraph,
    source_region: Union[str, RegionConfig, int],
    target_region: Union[str, RegionConfig, int],
) -> Tuple[str, str]:
    return (graph.region_name(source_region), graph.region_name(target_region))


def _canonical_eta_key(
    graph: RegionGraph,
    region_i: Union[str, RegionConfig, int],
    region_j: Union[str, RegionConfig, int],
) -> Tuple[str, str]:
    name_i = graph.region_name(region_i)
    name_j = graph.region_name(region_j)
    return (name_i, name_j) if name_i <= name_j else (name_j, name_i)


def _inverse_permutation(perm: Sequence[int]) -> Tuple[int, ...]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


# ============================================================
# Partial-trace adjoint
# ============================================================

def partial_trace_adjoint(
    omega: torch.Tensor,
    dims: Sequence[int],
    keep: Sequence[int],
) -> torch.Tensor:
    r"""
    Adjoint of the partial-trace map with respect to the Hilbert-Schmidt inner product.

    If
        T(A) = Tr_trace_out(A),
    where `keep` specifies the kept subsystem indices, then this function returns
    T^*(omega), characterized by
        <T(A), omega> = <A, T^*(omega)>.

    Construction
    ------------
    Let perm = keep + trace_out. In that permuted ordering,
        T^*(omega) = omega \otimes I_trace.
    Then we permute back to the original subsystem order.
    """
    dims = tuple(int(d) for d in dims)
    n = len(dims)
    keep = tuple(int(k) for k in keep)

    if len(set(keep)) != len(keep):
        raise ValueError(f"`keep` must not contain duplicates. Got {keep}.")
    if any(k < 0 or k >= n for k in keep):
        raise ValueError(f"`keep` must lie in [0, {n - 1}], got {keep}.")

    omega_t = _as_torch_tensor(omega)
    if omega_t.ndim != 2 or omega_t.shape[0] != omega_t.shape[1]:
        raise ValueError(f"omega must be a square matrix, got shape {tuple(omega_t.shape)}.")

    complex_dtype = _complex_dtype_for_tensor(omega_t)
    device = omega_t.device
    omega_t = _as_torch_tensor(omega_t, dtype=complex_dtype, device=device)

    trace_out = tuple(i for i in range(n) if i not in keep)
    dims_keep = tuple(dims[i] for i in keep)
    dims_trace = tuple(dims[i] for i in trace_out)

    dim_keep = int(torch.tensor(dims_keep).prod().item()) if len(dims_keep) > 0 else 1
    dim_trace = int(torch.tensor(dims_trace).prod().item()) if len(dims_trace) > 0 else 1

    if tuple(omega_t.shape) != (dim_keep, dim_keep):
        raise ValueError(
            f"omega has shape {tuple(omega_t.shape)}, expected {(dim_keep, dim_keep)} "
            f"for kept subsystem dimensions {dims_keep}."
        )

    if len(trace_out) == 0:
        return omega_t.clone()

    perm = tuple(keep) + tuple(trace_out)
    dims_perm = tuple(dims[i] for i in perm)

    ext_perm = torch.kron(
        omega_t,
        identity(dim_trace, dtype=complex_dtype, device=device),
    )
    inv_perm = _inverse_permutation(perm)
    ext_full = permute_subsystems(ext_perm, dims_perm, inv_perm)
    return hermitian_part(ext_full)


# ============================================================
# Consensus and dual initialization
# ============================================================

def initialize_eta_from_region_states(
    graph: RegionGraph,
    region_states: Mapping[str, torch.Tensor],
    average_pair_reductions: bool = True,
) -> Dict[Tuple[str, str], torch.Tensor]:
    """
    Initialize overlap consensus variables from a collection of regional states.
    """
    eta: Dict[Tuple[str, str], torch.Tensor] = {}

    for i, j in graph.overlap_pairs():
        name_i = graph.region_name(i)
        name_j = graph.region_name(j)
        key = (name_i, name_j)
        info = graph.overlap_info(i, j)

        red_i = partial_trace(
            region_states[name_i],
            dims=graph.region_site_dims(i),
            keep=info.local_keep_i,
        )
        red_j = partial_trace(
            region_states[name_j],
            dims=graph.region_site_dims(j),
            keep=info.local_keep_j,
        )

        if average_pair_reductions:
            eta[key] = hermitian_part(0.5 * (red_i + red_j))
        else:
            eta[key] = hermitian_part(red_i)

    return eta


def initialize_zero_duals(
    graph: RegionGraph,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[Tuple[str, str], torch.Tensor]:
    """
    Initialize all directed dual variables Lambda_{source,target} to zero.
    """
    device = None if device is None else torch.device(device)
    duals: Dict[Tuple[str, str], torch.Tensor] = {}

    for i, j in graph.overlap_pairs():
        name_i = graph.region_name(i)
        name_j = graph.region_name(j)
        dim_overlap = graph.overlap_dim(i, j)

        duals[(name_i, name_j)] = torch.zeros((dim_overlap, dim_overlap), dtype=dtype, device=device)
        duals[(name_j, name_i)] = torch.zeros((dim_overlap, dim_overlap), dtype=dtype, device=device)

    return duals


def validate_eta_collection(
    graph: RegionGraph,
    eta: Mapping[Tuple[str, str], torch.Tensor],
) -> None:
    """
    Validate eta collection against the region graph.
    """
    expected_keys = {
        (graph.region_name(i), graph.region_name(j))
        for i, j in graph.overlap_pairs()
    }
    provided_keys = set(eta.keys())

    missing = expected_keys - provided_keys
    extra = provided_keys - expected_keys

    if missing:
        raise ValueError(f"Missing eta keys: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected eta keys: {sorted(extra)}.")

    for i, j in graph.overlap_pairs():
        key = (graph.region_name(i), graph.region_name(j))
        expected_shape = (graph.overlap_dim(i, j), graph.overlap_dim(i, j))
        if tuple(eta[key].shape) != expected_shape:
            raise ValueError(
                f"eta[{key}] has shape {tuple(eta[key].shape)}, expected {expected_shape}."
            )


def validate_dual_collection(
    graph: RegionGraph,
    duals: Mapping[Tuple[str, str], torch.Tensor],
) -> None:
    """
    Validate directed dual collection against the region graph.
    """
    expected_keys = set()
    for i, j in graph.overlap_pairs():
        name_i = graph.region_name(i)
        name_j = graph.region_name(j)
        expected_keys.add((name_i, name_j))
        expected_keys.add((name_j, name_i))

    provided_keys = set(duals.keys())

    missing = expected_keys - provided_keys
    extra = provided_keys - expected_keys

    if missing:
        raise ValueError(f"Missing dual keys: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected dual keys: {sorted(extra)}.")

    for i, j in graph.overlap_pairs():
        dim_overlap = graph.overlap_dim(i, j)
        key_ij = (graph.region_name(i), graph.region_name(j))
        key_ji = (graph.region_name(j), graph.region_name(i))
        expected_shape = (dim_overlap, dim_overlap)

        if tuple(duals[key_ij].shape) != expected_shape:
            raise ValueError(
                f"duals[{key_ij}] has shape {tuple(duals[key_ij].shape)}, expected {expected_shape}."
            )
        if tuple(duals[key_ji].shape) != expected_shape:
            raise ValueError(
                f"duals[{key_ji}] has shape {tuple(duals[key_ji].shape)}, expected {expected_shape}."
            )


# ============================================================
# Region-local augmented objective and gradient
# ============================================================

def region_augmented_state_objective(
    graph: RegionGraph,
    region_name: str,
    rho: torch.Tensor,
    rho_outer_prev: torch.Tensor,
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

    shots = None if region_shots is None else int(region_shots[region_name])

    obj = state_subproblem_region_objective(
        empirical=empirical_probabilities[region_name],
        rho=rho,
        povm=region_povms[region_name],
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

        red = partial_trace(rho, dims=dims, keep=keep_self)
        resid = red - eta[eta_key]

        obj += hs_inner(duals[dual_key], resid)
        obj += 0.5 * beta * frobenius_norm(resid) ** 2

    return float(obj)


def region_augmented_state_gradient(
    graph: RegionGraph,
    region_name: str,
    rho: torch.Tensor,
    rho_outer_prev: torch.Tensor,
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
    Gradient of the region-local ADMM state-update objective with respect to rho.
    """
    beta = _ensure_positive_float(beta, "beta")
    gamma_rho = _ensure_positive_float(gamma_rho, "gamma_rho")

    shots = None if region_shots is None else int(region_shots[region_name])

    grad_parts = region_gradient_components(
        empirical=empirical_probabilities[region_name],
        rho=rho,
        povm=region_povms[region_name],
        confusion=fixed_confusions[region_name],
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )
    grad = _as_torch_tensor(grad_parts["grad_rho"]).clone()
    grad = grad + gamma_rho * (rho - rho_outer_prev)

    region_idx = graph.region_index(region_name)
    dims = graph.region_site_dims(region_idx)

    for neigh_idx in graph.neighbors(region_idx):
        eta_key = _canonical_eta_key(graph, region_idx, neigh_idx)
        dual_key = _directed_dual_key(graph, region_idx, neigh_idx)
        keep_self, _ = graph.local_keep_indices(region_idx, neigh_idx)

        red = partial_trace(rho, dims=dims, keep=keep_self)
        resid = red - eta[eta_key]
        overlap_term = duals[dual_key] + beta * resid
        grad = grad + partial_trace_adjoint(overlap_term, dims=dims, keep=keep_self)

    return hermitian_part(grad)


# ============================================================
# Region-local projected gradient solver
# ============================================================

@dataclass
class RegionStatePGInfo:
    converged: bool
    num_iters: int
    initial_objective: float
    final_objective: float
    final_grad_norm: float
    final_relative_change: float


def solve_region_state_update_pg(
    graph: RegionGraph,
    region_name: str,
    rho_init: torch.Tensor,
    rho_outer_prev: torch.Tensor,
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
    step_size: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-8,
    backtracking_factor: float = 0.5,
    armijo_c: float = 1e-4,
    max_backtracking_iters: int = 25,
) -> Tuple[torch.Tensor, RegionStatePGInfo]:
    """
    Solve one regional ADMM state update by projected gradient descent with backtracking.
    """
    step_size = _ensure_positive_float(step_size, "step_size")
    max_iters = _ensure_positive_int(max_iters, "max_iters")
    tol = _ensure_positive_float(tol, "tol")
    backtracking_factor = float(backtracking_factor)
    if not (0.0 < backtracking_factor < 1.0):
        raise ValueError(
            f"backtracking_factor must lie in (0,1), got {backtracking_factor}."
        )
    armijo_c = _ensure_nonnegative_float(armijo_c, "armijo_c")
    max_backtracking_iters = _ensure_positive_int(
        max_backtracking_iters,
        "max_backtracking_iters",
    )

    rho = project_to_density_matrix(rho_init)
    current_obj = region_augmented_state_objective(
        graph=graph,
        region_name=region_name,
        rho=rho,
        rho_outer_prev=rho_outer_prev,
        empirical_probabilities=empirical_probabilities,
        region_povms=region_povms,
        fixed_confusions=fixed_confusions,
        eta=eta,
        duals=duals,
        beta=beta,
        gamma_rho=gamma_rho,
        loss=loss,
        region_shots=region_shots,
        prob_floor=prob_floor,
    )
    initial_obj = current_obj

    converged = False
    final_grad_norm = float("nan")
    final_rel_change = float("nan")
    it_used = 0

    for it in range(1, max_iters + 1):
        grad = region_augmented_state_gradient(
            graph=graph,
            region_name=region_name,
            rho=rho,
            rho_outer_prev=rho_outer_prev,
            empirical_probabilities=empirical_probabilities,
            region_povms=region_povms,
            fixed_confusions=fixed_confusions,
            eta=eta,
            duals=duals,
            beta=beta,
            gamma_rho=gamma_rho,
            loss=loss,
            region_shots=region_shots,
            prob_floor=prob_floor,
        )
        grad_norm = frobenius_norm(grad)
        final_grad_norm = grad_norm
        it_used = it

        if grad_norm <= tol:
            converged = True
            final_rel_change = 0.0
            break

        accepted = False
        candidate_best = rho
        obj_best = current_obj
        step = step_size

        for _ in range(max_backtracking_iters):
            candidate = project_to_density_matrix(rho - step * grad)
            rel_change = frobenius_norm(candidate - rho) / max(frobenius_norm(rho), 1e-12)

            cand_obj = region_augmented_state_objective(
                graph=graph,
                region_name=region_name,
                rho=candidate,
                rho_outer_prev=rho_outer_prev,
                empirical_probabilities=empirical_probabilities,
                region_povms=region_povms,
                fixed_confusions=fixed_confusions,
                eta=eta,
                duals=duals,
                beta=beta,
                gamma_rho=gamma_rho,
                loss=loss,
                region_shots=region_shots,
                prob_floor=prob_floor,
            )

            sufficient_decrease_rhs = current_obj - armijo_c * (frobenius_norm(candidate - rho) ** 2) / max(step, 1e-12)
            if cand_obj <= sufficient_decrease_rhs + 1e-14:
                accepted = True
                candidate_best = candidate
                obj_best = cand_obj
                final_rel_change = rel_change
                break

            if cand_obj < obj_best - 1e-14:
                candidate_best = candidate
                obj_best = cand_obj
                final_rel_change = rel_change

            step *= backtracking_factor

        if not accepted:
            if obj_best < current_obj - 1e-14:
                rho = candidate_best
                current_obj = obj_best
            else:
                final_rel_change = 0.0
                break
        else:
            rho = candidate_best
            current_obj = obj_best

        if final_rel_change <= tol:
            converged = True
            break

    rho = project_to_density_matrix(rho)
    info = RegionStatePGInfo(
        converged=converged,
        num_iters=it_used,
        initial_objective=float(initial_obj),
        final_objective=float(current_obj),
        final_grad_norm=float(final_grad_norm),
        final_relative_change=float(final_rel_change),
    )
    return rho, info


# ============================================================
# ADMM result object
# ============================================================

@dataclass
class StateADMMResult:
    region_states: Dict[str, torch.Tensor]
    eta: Dict[Tuple[str, str], torch.Tensor]
    duals: Dict[Tuple[str, str], torch.Tensor]
    converged: bool
    num_iterations: int
    final_primal_residual: float
    final_dual_residual: float
    final_max_overlap_residual: float
    history: Dict[str, List[float]]

    def validate(self, cfg: ExperimentConfig, graph: Optional[RegionGraph] = None) -> None:
        """
        Validate the ADMM result structure.
        """
        validate_region_state_collection(cfg, self.region_states, check_overlap_consistency=False)

        graph = RegionGraph(cfg) if graph is None else graph
        validate_eta_collection(graph, self.eta)
        validate_dual_collection(graph, self.duals)

    def pretty_print(self) -> None:
        print("=" * 72)
        print("StateADMMResult")
        print("-" * 72)
        print(f"Converged: {self.converged}")
        print(f"Iterations: {self.num_iterations}")
        print(f"Final primal residual: {self.final_primal_residual:.6e}")
        print(f"Final dual residual: {self.final_dual_residual:.6e}")
        print(f"Final max overlap residual: {self.final_max_overlap_residual:.6e}")
        print("=" * 72)


# ============================================================
# Main ADMM solver
# ============================================================

def solve_state_subproblem_admm(
    cfg: ExperimentConfig,
    empirical_probabilities: Mapping[str, torch.Tensor],
    region_povms: Mapping[str, POVM],
    fixed_confusions: Mapping[str, torch.Tensor],
    region_states_outer_prev: Mapping[str, torch.Tensor],
    eta_init: Optional[Mapping[Tuple[str, str], torch.Tensor]] = None,
    duals_init: Optional[Mapping[Tuple[str, str], torch.Tensor]] = None,
    *,
    graph: Optional[RegionGraph] = None,
    loss: Optional[Union[str, LossConfig]] = None,
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
    beta: Optional[float] = None,
    gamma_rho: Optional[float] = None,
    inner_max_iters: Optional[int] = None,
    inner_primal_tol: Optional[float] = None,
    inner_dual_tol: Optional[float] = None,
    state_step_size: Optional[float] = None,
    state_gd_max_iters: Optional[int] = None,
    state_gd_tol: Optional[float] = None,
    verbose: Optional[bool] = None,
) -> StateADMMResult:
    """
    Solve the proximal state subproblem by inner ADMM.
    """
    graph = RegionGraph(cfg) if graph is None else graph
    loss = _resolve_loss(loss, cfg)
    prob_floor = _resolve_prob_floor(loss, prob_floor)

    beta = cfg.admm.beta if beta is None else beta
    gamma_rho = cfg.admm.gamma_rho if gamma_rho is None else gamma_rho
    inner_max_iters = cfg.admm.inner_max_iters if inner_max_iters is None else inner_max_iters
    inner_primal_tol = cfg.admm.inner_primal_tol if inner_primal_tol is None else inner_primal_tol
    inner_dual_tol = cfg.admm.inner_dual_tol if inner_dual_tol is None else inner_dual_tol
    state_step_size = cfg.admm.state_step_size if state_step_size is None else state_step_size
    state_gd_max_iters = cfg.admm.state_gd_max_iters if state_gd_max_iters is None else state_gd_max_iters
    state_gd_tol = cfg.admm.state_gd_tol if state_gd_tol is None else state_gd_tol
    verbose = cfg.admm.verbose if verbose is None else bool(verbose)

    beta = _ensure_positive_float(beta, "beta")
    gamma_rho = _ensure_positive_float(gamma_rho, "gamma_rho")
    inner_max_iters = _ensure_positive_int(inner_max_iters, "inner_max_iters")
    inner_primal_tol = _ensure_positive_float(inner_primal_tol, "inner_primal_tol")
    inner_dual_tol = _ensure_positive_float(inner_dual_tol, "inner_dual_tol")
    state_step_size = _ensure_positive_float(state_step_size, "state_step_size")
    state_gd_max_iters = _ensure_positive_int(state_gd_max_iters, "state_gd_max_iters")
    state_gd_tol = _ensure_positive_float(state_gd_tol, "state_gd_tol")

    validate_region_state_collection(cfg, dict(region_states_outer_prev), check_overlap_consistency=False)

    expected_names = {region.name for region in cfg.regions}
    if set(empirical_probabilities.keys()) != expected_names:
        raise ValueError("empirical_probabilities keys must match cfg.regions.")
    if set(region_povms.keys()) != expected_names:
        raise ValueError("region_povms keys must match cfg.regions.")
    if set(fixed_confusions.keys()) != expected_names:
        raise ValueError("fixed_confusions keys must match cfg.regions.")

    if region_shots is None:
        region_shots = build_region_shot_dict(cfg)

    rho_curr = _copy_matrix_dict(region_states_outer_prev)

    if eta_init is None:
        eta_curr = initialize_eta_from_region_states(graph, rho_curr, average_pair_reductions=True)
    else:
        eta_curr = _copy_matrix_dict(eta_init)
        validate_eta_collection(graph, eta_curr)

    if duals_init is None:
        ref_tensor = next(iter(rho_curr.values()))
        duals_curr = initialize_zero_duals(
            graph,
            dtype=_complex_dtype_for_tensor(ref_tensor),
            device=ref_tensor.device,
        )
    else:
        duals_curr = _copy_matrix_dict(duals_init)
        validate_dual_collection(graph, duals_curr)

    history: Dict[str, List[float]] = {
        "primal_residual": [],
        "dual_residual": [],
        "max_overlap_residual": [],
        "average_region_pg_iters": [],
    }

    converged = False
    final_primal = float("nan")
    final_dual = float("nan")
    final_max_overlap = float("nan")
    it_used = 0

    overlap_pairs = graph.overlap_pairs()
    if len(overlap_pairs) == 0:
        rho_new: Dict[str, torch.Tensor] = {}
        pg_iters = []
        for region in cfg.regions:
            name = region.name
            rho_new[name], pg_info = solve_region_state_update_pg(
                graph=graph,
                region_name=name,
                rho_init=rho_curr[name],
                rho_outer_prev=region_states_outer_prev[name],
                empirical_probabilities=empirical_probabilities,
                region_povms=region_povms,
                fixed_confusions=fixed_confusions,
                eta=eta_curr,
                duals=duals_curr,
                beta=beta,
                gamma_rho=gamma_rho,
                loss=loss,
                region_shots=region_shots,
                prob_floor=prob_floor,
                step_size=state_step_size,
                max_iters=state_gd_max_iters,
                tol=state_gd_tol,
            )
            pg_iters.append(pg_info.num_iters)

        result = StateADMMResult(
            region_states=rho_new,
            eta=eta_curr,
            duals=duals_curr,
            converged=True,
            num_iterations=1,
            final_primal_residual=0.0,
            final_dual_residual=0.0,
            final_max_overlap_residual=0.0,
            history={
                "primal_residual": [0.0],
                "dual_residual": [0.0],
                "max_overlap_residual": [0.0],
                "average_region_pg_iters": [float(sum(pg_iters) / len(pg_iters)) if pg_iters else 0.0],
            },
        )
        result.validate(cfg, graph)
        return result

    for it in range(1, inner_max_iters + 1):
        it_used = it

        # 1) Region-state updates
        rho_next: Dict[str, torch.Tensor] = {}
        region_pg_iters: List[float] = []

        for region in cfg.regions:
            name = region.name
            rho_updated, pg_info = solve_region_state_update_pg(
                graph=graph,
                region_name=name,
                rho_init=rho_curr[name],
                rho_outer_prev=region_states_outer_prev[name],
                empirical_probabilities=empirical_probabilities,
                region_povms=region_povms,
                fixed_confusions=fixed_confusions,
                eta=eta_curr,
                duals=duals_curr,
                beta=beta,
                gamma_rho=gamma_rho,
                loss=loss,
                region_shots=region_shots,
                prob_floor=prob_floor,
                step_size=state_step_size,
                max_iters=state_gd_max_iters,
                tol=state_gd_tol,
            )
            rho_next[name] = rho_updated
            region_pg_iters.append(float(pg_info.num_iters))

        # 2) Consensus variable updates
        eta_prev = _copy_matrix_dict(eta_curr)
        eta_next: Dict[Tuple[str, str], torch.Tensor] = {}

        for i, j in overlap_pairs:
            name_i = graph.region_name(i)
            name_j = graph.region_name(j)
            key = (name_i, name_j)
            info = graph.overlap_info(i, j)

            red_i = partial_trace(
                rho_next[name_i],
                dims=graph.region_site_dims(i),
                keep=info.local_keep_i,
            )
            red_j = partial_trace(
                rho_next[name_j],
                dims=graph.region_site_dims(j),
                keep=info.local_keep_j,
            )

            dual_ij = duals_curr[(name_i, name_j)]
            dual_ji = duals_curr[(name_j, name_i)]

            eta_val = 0.5 * (red_i + red_j + (dual_ij + dual_ji) / beta)
            eta_next[key] = hermitian_part(eta_val)

        # 3) Dual updates
        duals_next: Dict[Tuple[str, str], torch.Tensor] = {}

        for i, j in overlap_pairs:
            name_i = graph.region_name(i)
            name_j = graph.region_name(j)
            key = (name_i, name_j)
            info = graph.overlap_info(i, j)

            red_i = partial_trace(
                rho_next[name_i],
                dims=graph.region_site_dims(i),
                keep=info.local_keep_i,
            )
            red_j = partial_trace(
                rho_next[name_j],
                dims=graph.region_site_dims(j),
                keep=info.local_keep_j,
            )

            eta_ij = eta_next[key]

            duals_next[(name_i, name_j)] = hermitian_part(
                duals_curr[(name_i, name_j)] + beta * (red_i - eta_ij)
            )
            duals_next[(name_j, name_i)] = hermitian_part(
                duals_curr[(name_j, name_i)] + beta * (red_j - eta_ij)
            )

        # 4) Residuals and stopping
        final_primal = overlap_primal_residual_norm(graph, rho_next, eta_next)
        final_dual = overlap_dual_residual_norm(beta, eta_next, eta_prev)
        final_max_overlap = max_overlap_residual(graph, rho_next, eta_next)

        history["primal_residual"].append(float(final_primal))
        history["dual_residual"].append(float(final_dual))
        history["max_overlap_residual"].append(float(final_max_overlap))
        history["average_region_pg_iters"].append(
            float(sum(region_pg_iters) / len(region_pg_iters)) if region_pg_iters else 0.0
        )

        if verbose:
            print(
                f"[state_admm] iter={it:03d} "
                f"primal={final_primal:.6e} "
                f"dual={final_dual:.6e} "
                f"max_overlap={final_max_overlap:.6e} "
                f"avg_pg_iters={history['average_region_pg_iters'][-1]:.2f}"
            )

        rho_curr = rho_next
        eta_curr = eta_next
        duals_curr = duals_next

        if final_primal <= inner_primal_tol and final_dual <= inner_dual_tol:
            converged = True
            break

    result = StateADMMResult(
        region_states=rho_curr,
        eta=eta_curr,
        duals=duals_curr,
        converged=converged,
        num_iterations=it_used,
        final_primal_residual=float(final_primal),
        final_dual_residual=float(final_dual),
        final_max_overlap_residual=float(final_max_overlap),
        history=history,
    )
    result.validate(cfg, graph)
    return result


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_partial_trace_adjoint_identity() -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)

    dims = (2, 3, 2)
    dim_full = int(torch.tensor(dims).prod().item())
    keep = (0, 2)

    a = torch.randn(dim_full, dim_full, dtype=torch.float64, generator=gen) + 1j * torch.randn(
        dim_full, dim_full, dtype=torch.float64, generator=gen
    )
    a = hermitian_part(a)

    dim_keep = dims[0] * dims[2]
    omega = torch.randn(dim_keep, dim_keep, dtype=torch.float64, generator=gen) + 1j * torch.randn(
        dim_keep, dim_keep, dtype=torch.float64, generator=gen
    )
    omega = hermitian_part(omega)

    lhs = hs_inner(partial_trace(a, dims=dims, keep=keep), omega)
    rhs = hs_inner(a, partial_trace_adjoint(omega, dims=dims, keep=keep))

    assert abs(lhs - rhs) <= 1e-10


def _self_test_initializers() -> None:
    from config import make_default_experiment_config
    from simulator import simulate_experiment

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    sim = simulate_experiment(cfg, truth_mode="global_consistent")
    graph = RegionGraph(cfg)

    eta = initialize_eta_from_region_states(graph, sim.region_states)
    duals = initialize_zero_duals(
        graph,
        dtype=cfg.torch_complex_dtype,
        device=cfg.device,
    )

    validate_eta_collection(graph, eta)
    validate_dual_collection(graph, duals)

    for key, val in eta.items():
        assert val.shape[0] == val.shape[1]
    for key, val in duals.items():
        assert torch.allclose(val, torch.zeros_like(val))


def _self_test_state_admm_truth_fixed_point() -> None:
    from config import make_default_experiment_config
    from simulator import simulate_experiment

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    sim = simulate_experiment(cfg, truth_mode="global_consistent")
    graph = RegionGraph(cfg)

    eta0 = initialize_eta_from_region_states(graph, sim.region_states)
    dual0 = initialize_zero_duals(
        graph,
        dtype=cfg.torch_complex_dtype,
        device=cfg.device,
    )

    result = solve_state_subproblem_admm(
        cfg=cfg,
        empirical_probabilities=sim.empirical_probabilities,
        region_povms=sim.region_povms,
        fixed_confusions=sim.region_confusions,
        region_states_outer_prev=sim.region_states,
        eta_init=eta0,
        duals_init=dual0,
        graph=graph,
        loss="l2",
        region_shots=sim.region_shots,
        beta=1.0,
        gamma_rho=1.0,
        inner_max_iters=5,
        inner_primal_tol=1e-10,
        inner_dual_tol=1e-10,
        state_step_size=0.1,
        state_gd_max_iters=50,
        state_gd_tol=1e-10,
        verbose=False,
    )

    result.validate(cfg, graph)

    assert result.final_primal_residual <= 1e-8
    assert result.final_dual_residual <= 1e-8

    for region in cfg.regions:
        name = region.name
        assert is_density_matrix(result.region_states[name])
        assert frobenius_norm(result.region_states[name] - sim.region_states[name]) <= 1e-7


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the state_admm module.
    """
    tests = [
        ("partial-trace adjoint identity", _self_test_partial_trace_adjoint_identity),
        ("eta and dual initialization", _self_test_initializers),
        ("truth fixed-point ADMM solve", _self_test_state_admm_truth_fixed_point),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All state_admm self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
