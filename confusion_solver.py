from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch

from config import ExperimentConfig, LossConfig
from measurements import POVM, measurement_map
from noise import (
    apply_confusion_matrix,
    build_all_reference_confusions,
    project_confusion_matrix,
    validate_confusion_matrix,
    validate_region_confusion_collection,
)
from objectives import (
    build_region_shot_dict,
    confusion_subproblem_region_objective,
    gradient_wrt_predicted,
)
from states import validate_region_state_collection


# ============================================================
# Small helpers
# ============================================================

def _coerce_device(device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _as_torch_tensor(x, dtype=None, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(
            dtype=x.dtype if dtype is None else dtype,
            device=x.device if device is None else device,
        )
    return torch.as_tensor(x, dtype=dtype, device=device)


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


def _copy_matrix_dict(d: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: _as_torch_tensor(v).clone() for k, v in d.items()}


def _relative_fro_change(current: torch.Tensor, previous: torch.Tensor, eps: float = 1e-12) -> float:
    num = float(torch.linalg.matrix_norm(current - previous, ord="fro").item())
    den = max(float(torch.linalg.matrix_norm(previous, ord="fro").item()), eps)
    return num / den


def _matrix_fro_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(x, ord="fro").item())


def _real_dtype_from_tensor(x: torch.Tensor) -> torch.dtype:
    if x.dtype == torch.float32:
        return torch.float32
    if x.dtype == torch.float64:
        return torch.float64
    if x.dtype == torch.complex64:
        return torch.float32
    if x.dtype == torch.complex128:
        return torch.float64
    raise ValueError(f"Unsupported dtype: {x.dtype}.")


# ============================================================
# Region-local objective and gradient
# ============================================================

def confusion_region_gradient(
    empirical: torch.Tensor,
    ideal_probabilities: torch.Tensor,
    confusion: torch.Tensor,
    reference_confusion: torch.Tensor,
    lambda_confusion: float,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
    gamma_c: float = 0.0,
    confusion_prev: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    Gradient of the local confusion-update objective:
        D(\hat p, C p_ideal)
        + lambda ||C - C_ref||_F^2
        + (gamma_c / 2) ||C - C_prev||_F^2.

    If
        q = C p_ideal
        g_q = \nabla_q D(\hat p, q),
    then
        grad_C(data) = g_q p_ideal^T.

    Regularization gradients:
        grad_C(reg)  = 2 lambda (C - C_ref)
        grad_C(prox) = gamma_c (C - C_prev)
    """
    lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")
    gamma_c = _ensure_nonnegative_float(gamma_c, "gamma_c")

    confusion_t = _as_torch_tensor(confusion)
    real_dtype = _real_dtype_from_tensor(confusion_t)
    device = confusion_t.device

    empirical_t = _as_torch_tensor(empirical, dtype=real_dtype, device=device).reshape(-1)
    ideal_t = _as_torch_tensor(ideal_probabilities, dtype=real_dtype, device=device).reshape(-1)
    confusion_t = _as_torch_tensor(confusion, dtype=real_dtype, device=device)
    reference_t = _as_torch_tensor(reference_confusion, dtype=real_dtype, device=device)

    if confusion_t.ndim != 2 or confusion_t.shape[0] != confusion_t.shape[1]:
        raise ValueError(f"confusion must be square, got shape {tuple(confusion_t.shape)}.")
    if reference_t.shape != confusion_t.shape:
        raise ValueError(
            f"reference_confusion must match confusion shape, got {tuple(reference_t.shape)} "
            f"and {tuple(confusion_t.shape)}."
        )
    if ideal_t.numel() != confusion_t.shape[1]:
        raise ValueError(
            f"ideal_probabilities has length {ideal_t.numel()}, but confusion expects "
            f"{confusion_t.shape[1]} outcomes."
        )
    if empirical_t.numel() != confusion_t.shape[0]:
        raise ValueError(
            f"empirical has length {empirical_t.numel()}, but confusion outputs "
            f"{confusion_t.shape[0]} outcomes."
        )

    noisy = apply_confusion_matrix(confusion_t, ideal_t, prob_floor=0.0)
    grad_q = gradient_wrt_predicted(
        empirical=empirical_t,
        predicted=noisy,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    ).to(dtype=real_dtype, device=device)

    grad = torch.outer(grad_q, ideal_t)
    grad = grad + 2.0 * lambda_confusion * (confusion_t - reference_t)

    if gamma_c > 0.0:
        if confusion_prev is None:
            raise ValueError("confusion_prev must be provided when gamma_c > 0.")
        confusion_prev_t = _as_torch_tensor(confusion_prev, dtype=real_dtype, device=device)
        if confusion_prev_t.shape != confusion_t.shape:
            raise ValueError(
                f"confusion_prev must match confusion shape, got {tuple(confusion_prev_t.shape)} "
                f"and {tuple(confusion_t.shape)}."
            )
        grad = grad + gamma_c * (confusion_t - confusion_prev_t)

    return grad


# ============================================================
# Single-region projected-gradient solver
# ============================================================

@dataclass
class RegionConfusionPGInfo:
    converged: bool
    num_iters: int
    initial_objective: float
    final_objective: float
    final_grad_norm: float
    final_relative_change: float


def solve_region_confusion_update_pg(
    empirical: torch.Tensor,
    ideal_probabilities: torch.Tensor,
    confusion_init: torch.Tensor,
    reference_confusion: torch.Tensor,
    lambda_confusion: float,
    loss: Union[str, LossConfig],
    *,
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
    gamma_c: float = 0.0,
    confusion_prev: Optional[torch.Tensor] = None,
    step_size: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-8,
    max_backtracking_iters: int = 25,
    backtracking_factor: float = 0.5,
    armijo_c: float = 1e-4,
) -> Tuple[torch.Tensor, RegionConfusionPGInfo]:
    """
    Solve one region-local confusion update by projected gradient descent.

    The feasible set is the set of column-stochastic matrices. Each gradient step
    is followed by projection onto that set.
    """
    lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")
    gamma_c = _ensure_nonnegative_float(gamma_c, "gamma_c")
    step_size = _ensure_positive_float(step_size, "step_size")
    max_iters = _ensure_positive_int(max_iters, "max_iters")
    tol = _ensure_positive_float(tol, "tol")
    max_backtracking_iters = _ensure_positive_int(max_backtracking_iters, "max_backtracking_iters")

    confusion_t = _as_torch_tensor(confusion_init)
    real_dtype = _real_dtype_from_tensor(confusion_t)
    device = confusion_t.device

    empirical_t = _as_torch_tensor(empirical, dtype=real_dtype, device=device).reshape(-1)
    ideal_t = _as_torch_tensor(ideal_probabilities, dtype=real_dtype, device=device).reshape(-1)
    reference_t = _as_torch_tensor(reference_confusion, dtype=real_dtype, device=device)
    confusion_t = _as_torch_tensor(confusion_init, dtype=real_dtype, device=device)

    if gamma_c > 0.0:
        if confusion_prev is None:
            raise ValueError("confusion_prev must be provided when gamma_c > 0.")
        confusion_prev_t = _as_torch_tensor(confusion_prev, dtype=real_dtype, device=device)
    else:
        confusion_prev_t = None

    validate_confusion_matrix(confusion_t, num_outcomes=ideal_t.numel())
    validate_confusion_matrix(reference_t, num_outcomes=ideal_t.numel())
    if confusion_prev_t is not None:
        validate_confusion_matrix(confusion_prev_t, num_outcomes=ideal_t.numel())

    confusion_t = project_confusion_matrix(confusion_t)
    initial_obj = confusion_subproblem_region_objective(
        empirical=empirical_t,
        ideal_probabilities=ideal_t,
        confusion=confusion_t,
        reference_confusion=reference_t,
        lambda_confusion=lambda_confusion,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
        gamma_c=gamma_c,
        confusion_prev=confusion_prev_t,
    )
    current_obj = float(initial_obj)

    converged = False
    it_used = 0
    final_grad_norm = float("inf")
    final_rel_change = float("inf")

    for it in range(1, max_iters + 1):
        grad = confusion_region_gradient(
            empirical=empirical_t,
            ideal_probabilities=ideal_t,
            confusion=confusion_t,
            reference_confusion=reference_t,
            lambda_confusion=lambda_confusion,
            loss=loss,
            shots=shots,
            prob_floor=prob_floor,
            gamma_c=gamma_c,
            confusion_prev=confusion_prev_t,
        )
        grad_norm = _matrix_fro_norm(grad)
        final_grad_norm = grad_norm
        it_used = it

        if grad_norm <= tol:
            converged = True
            final_rel_change = 0.0
            break

        accepted = False
        step = float(step_size)
        candidate_best = confusion_t
        obj_best = current_obj

        for _ in range(max_backtracking_iters):
            candidate = project_confusion_matrix(confusion_t - step * grad)
            rel_change = _relative_fro_change(candidate, confusion_t)

            cand_obj = confusion_subproblem_region_objective(
                empirical=empirical_t,
                ideal_probabilities=ideal_t,
                confusion=candidate,
                reference_confusion=reference_t,
                lambda_confusion=lambda_confusion,
                loss=loss,
                shots=shots,
                prob_floor=prob_floor,
                gamma_c=gamma_c,
                confusion_prev=confusion_prev_t,
            )

            sufficient_decrease_rhs = current_obj - armijo_c * (
                _matrix_fro_norm(candidate - confusion_t) ** 2
            ) / max(step, 1e-12)

            if cand_obj <= sufficient_decrease_rhs + 1e-14:
                accepted = True
                candidate_best = candidate
                obj_best = float(cand_obj)
                final_rel_change = float(rel_change)
                break

            if cand_obj < obj_best - 1e-14:
                candidate_best = candidate
                obj_best = float(cand_obj)
                final_rel_change = float(rel_change)

            step *= backtracking_factor

        if not accepted:
            if obj_best < current_obj - 1e-14:
                confusion_t = candidate_best
                current_obj = obj_best
            else:
                final_rel_change = 0.0
                break
        else:
            confusion_t = candidate_best
            current_obj = obj_best

        if final_rel_change <= tol:
            converged = True
            break

    confusion_t = project_confusion_matrix(confusion_t)
    info = RegionConfusionPGInfo(
        converged=converged,
        num_iters=it_used,
        initial_objective=float(initial_obj),
        final_objective=float(current_obj),
        final_grad_norm=float(final_grad_norm),
        final_relative_change=float(final_rel_change),
    )
    return confusion_t, info


# ============================================================
# Batch solver and result object
# ============================================================

@dataclass
class ConfusionUpdateResult:
    region_confusions: Dict[str, torch.Tensor]
    converged: bool
    average_pg_iters: float
    max_pg_iters: int
    history: Dict[str, List[float]]

    def validate(self, cfg: ExperimentConfig) -> None:
        validate_region_confusion_collection(cfg, self.region_confusions)

    def pretty_print(self) -> None:
        print("=" * 72)
        print("ConfusionUpdateResult")
        print("-" * 72)
        print(f"Converged: {self.converged}")
        print(f"Average PG iterations: {self.average_pg_iters:.2f}")
        print(f"Max PG iterations: {self.max_pg_iters}")
        print("=" * 72)


def update_all_confusions(
    cfg: ExperimentConfig,
    empirical_probabilities: Mapping[str, torch.Tensor],
    region_states_fixed: Mapping[str, torch.Tensor],
    region_povms: Mapping[str, POVM],
    confusion_prev: Mapping[str, torch.Tensor],
    reference_confusions: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    loss: Optional[Union[str, LossConfig]] = None,
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
    lambda_confusion: Optional[float] = None,
    gamma_c: Optional[float] = None,
    step_size: Optional[float] = None,
    max_iters: Optional[int] = None,
    tol: Optional[float] = None,
    verbose: Optional[bool] = None,
) -> ConfusionUpdateResult:
    """
    Update all regional confusion matrices independently for fixed regional states.

    Parameters
    ----------
    cfg :
        Experiment configuration.
    empirical_probabilities :
        Region-name -> empirical frequency vector.
    region_states_fixed :
        Region-name -> fixed regional state rho^{k+1}.
    region_povms :
        Region-name -> POVM.
    confusion_prev :
        Region-name -> previous confusion matrix C^k.
    reference_confusions :
        Region-name -> reference confusion matrix. If None, built from cfg.
    loss :
        Optional loss override. If None, use cfg.loss.
    region_shots :
        Optional region-name -> shots mapping. If None, use cfg.
    prob_floor :
        Optional probability floor override for NLL.
    lambda_confusion, gamma_c, step_size, max_iters, tol, verbose :
        Optional overrides of cfg.admm settings.

    Returns
    -------
    ConfusionUpdateResult
        Updated confusion matrices and diagnostics.
    """
    loss = _resolve_loss(loss, cfg)
    prob_floor = _resolve_prob_floor(loss, prob_floor)

    lambda_confusion = (
        cfg.admm.lambda_confusion if lambda_confusion is None else lambda_confusion
    )
    gamma_c = cfg.admm.gamma_c if gamma_c is None else gamma_c
    step_size = cfg.admm.confusion_step_size if step_size is None else step_size
    max_iters = cfg.admm.confusion_gd_max_iters if max_iters is None else max_iters
    tol = cfg.admm.confusion_gd_tol if tol is None else tol
    verbose = cfg.admm.verbose if verbose is None else bool(verbose)

    lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")
    gamma_c = _ensure_positive_float(gamma_c, "gamma_c")
    step_size = _ensure_positive_float(step_size, "step_size")
    max_iters = _ensure_positive_int(max_iters, "max_iters")
    tol = _ensure_positive_float(tol, "tol")

    validate_region_state_collection(cfg, dict(region_states_fixed), check_overlap_consistency=False)
    validate_region_confusion_collection(cfg, dict(confusion_prev))

    expected_names = {region.name for region in cfg.regions}
    if set(empirical_probabilities.keys()) != expected_names:
        raise ValueError("empirical_probabilities keys must match cfg.regions.")
    if set(region_povms.keys()) != expected_names:
        raise ValueError("region_povms keys must match cfg.regions.")

    if reference_confusions is None:
        reference_confusions = build_all_reference_confusions(cfg)
    else:
        validate_region_confusion_collection(cfg, dict(reference_confusions))

    if region_shots is None:
        region_shots = build_region_shot_dict(cfg)

    updated_confusions: Dict[str, torch.Tensor] = {}
    per_region_iters: List[int] = []
    per_region_converged: List[bool] = []
    initial_objectives: List[float] = []
    final_objectives: List[float] = []
    final_grad_norms: List[float] = []
    final_relative_changes: List[float] = []

    for region in cfg.regions:
        name = region.name
        rho = region_states_fixed[name]
        povm = region_povms[name]
        ideal_probabilities = measurement_map(rho, povm)

        c_new, info = solve_region_confusion_update_pg(
            empirical=empirical_probabilities[name],
            ideal_probabilities=ideal_probabilities,
            confusion_init=confusion_prev[name],
            reference_confusion=reference_confusions[name],
            lambda_confusion=lambda_confusion,
            loss=loss,
            shots=int(region_shots[name]),
            prob_floor=prob_floor,
            gamma_c=gamma_c,
            confusion_prev=confusion_prev[name],
            step_size=step_size,
            max_iters=max_iters,
            tol=tol,
        )

        updated_confusions[name] = c_new
        per_region_iters.append(info.num_iters)
        per_region_converged.append(info.converged)
        initial_objectives.append(info.initial_objective)
        final_objectives.append(info.final_objective)
        final_grad_norms.append(info.final_grad_norm)
        final_relative_changes.append(info.final_relative_change)

        if verbose:
            print(
                f"[confusion_solver] region={name} "
                f"iters={info.num_iters:03d} "
                f"conv={info.converged} "
                f"obj0={info.initial_objective:.6e} "
                f"objf={info.final_objective:.6e} "
                f"grad={info.final_grad_norm:.6e}"
            )

    result = ConfusionUpdateResult(
        region_confusions=updated_confusions,
        converged=all(per_region_converged),
        average_pg_iters=float(sum(per_region_iters) / len(per_region_iters)) if per_region_iters else 0.0,
        max_pg_iters=max(per_region_iters) if per_region_iters else 0,
        history={
            "per_region_iters": [float(v) for v in per_region_iters],
            "initial_objectives": [float(v) for v in initial_objectives],
            "final_objectives": [float(v) for v in final_objectives],
            "final_grad_norms": [float(v) for v in final_grad_norms],
            "final_relative_changes": [float(v) for v in final_relative_changes],
        },
    )
    result.validate(cfg)
    return result


# ============================================================
# Lightweight self-tests
# ============================================================

def _finite_difference_check_confusion_gradient_l2() -> None:
    empirical = torch.tensor([0.55, 0.45], dtype=torch.float64)
    ideal = torch.tensor([0.7, 0.3], dtype=torch.float64)

    c = torch.tensor([[0.90, 0.10], [0.10, 0.90]], dtype=torch.float64)
    c = project_confusion_matrix(c)
    ref = torch.eye(2, dtype=torch.float64)

    grad = confusion_region_gradient(
        empirical=empirical,
        ideal_probabilities=ideal,
        confusion=c,
        reference_confusion=ref,
        lambda_confusion=0.3,
        loss="l2",
        shots=None,
        gamma_c=0.2,
        confusion_prev=c,
    )

    eps = 1e-7

    # IMPORTANT:
    # This direction preserves column sums exactly, so c ± eps*direction
    # remain in the feasible affine set of column-stochastic matrices
    # for sufficiently small eps.
    direction = torch.tensor(
        [
            [0.30, -0.40],
            [-0.30, 0.40],
        ],
        dtype=torch.float64,
    )
    direction = direction / torch.linalg.matrix_norm(direction, ord="fro")

    c_plus = c + eps * direction
    c_minus = c - eps * direction

    validate_confusion_matrix(c_plus, num_outcomes=2)
    validate_confusion_matrix(c_minus, num_outcomes=2)

    f_plus = confusion_subproblem_region_objective(
        empirical=empirical,
        ideal_probabilities=ideal,
        confusion=c_plus,
        reference_confusion=ref,
        lambda_confusion=0.3,
        loss="l2",
        gamma_c=0.2,
        confusion_prev=c,
    )
    f_minus = confusion_subproblem_region_objective(
        empirical=empirical,
        ideal_probabilities=ideal,
        confusion=c_minus,
        reference_confusion=ref,
        lambda_confusion=0.3,
        loss="l2",
        gamma_c=0.2,
        confusion_prev=c,
    )

    fd = (f_plus - f_minus) / (2.0 * eps)
    ip = float(torch.sum(grad * direction).item())

    assert abs(fd - ip) <= 1e-5


def _self_test_region_confusion_solver_fixed_point() -> None:
    empirical = torch.tensor([0.72, 0.28], dtype=torch.float64)
    ideal = torch.tensor([0.72, 0.28], dtype=torch.float64)

    c0 = torch.eye(2, dtype=torch.float64)
    ref = torch.eye(2, dtype=torch.float64)

    c_new, info = solve_region_confusion_update_pg(
        empirical=empirical,
        ideal_probabilities=ideal,
        confusion_init=c0,
        reference_confusion=ref,
        lambda_confusion=0.1,
        loss="l2",
        gamma_c=1.0,
        confusion_prev=c0,
        step_size=0.2,
        max_iters=50,
        tol=1e-10,
    )

    validate_confusion_matrix(c_new, num_outcomes=2)
    assert _matrix_fro_norm(c_new - c0) <= 1e-8
    assert info.final_objective <= info.initial_objective + 1e-12


def _self_test_batch_update_shapes() -> None:
    from config import make_default_experiment_config
    from simulator import simulate_experiment
    from noise import build_all_initial_confusions

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    sim = simulate_experiment(cfg, truth_mode="global_consistent")
    init_confusions = build_all_initial_confusions(cfg)

    result = update_all_confusions(
        cfg=cfg,
        empirical_probabilities=sim.empirical_probabilities,
        region_states_fixed=sim.region_states,
        region_povms=sim.region_povms,
        confusion_prev=init_confusions,
        reference_confusions=build_all_reference_confusions(cfg),
        loss="l2",
        region_shots=sim.region_shots,
        lambda_confusion=cfg.admm.lambda_confusion,
        gamma_c=cfg.admm.gamma_c,
        step_size=0.1,
        max_iters=20,
        tol=1e-8,
        verbose=False,
    )

    result.validate(cfg)
    assert set(result.region_confusions.keys()) == {region.name for region in cfg.regions}
    for region in cfg.regions:
        c = result.region_confusions[region.name]
        validate_confusion_matrix(c)


def _self_test_truth_fixed_point_identity_case() -> None:
    from config import make_default_experiment_config
    from simulator import simulate_experiment

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    # Force true confusion = identity and initial = identity
    for region in cfg.regions:
        region.true_confusion_model = "identity"
        region.init_confusion_method = "identity"

    sim = simulate_experiment(cfg, truth_mode="global_consistent")
    result = update_all_confusions(
        cfg=cfg,
        empirical_probabilities=sim.empirical_probabilities,
        region_states_fixed=sim.region_states,
        region_povms=sim.region_povms,
        confusion_prev=sim.region_confusions,
        reference_confusions=build_all_reference_confusions(cfg),
        loss="l2",
        region_shots=sim.region_shots,
        lambda_confusion=cfg.admm.lambda_confusion,
        gamma_c=cfg.admm.gamma_c,
        step_size=0.1,
        max_iters=50,
        tol=1e-10,
        verbose=False,
    )

    result.validate(cfg)
    for region in cfg.regions:
        name = region.name
        assert _matrix_fro_norm(result.region_confusions[name] - sim.region_confusions[name]) <= 1e-7


def _self_test_gpu_smoke() -> None:
    if not torch.cuda.is_available():
        return

    empirical = torch.tensor([0.55, 0.45], dtype=torch.float64, device="cuda")
    ideal = torch.tensor([0.7, 0.3], dtype=torch.float64, device="cuda")
    c0 = torch.tensor([[0.90, 0.10], [0.10, 0.90]], dtype=torch.float64, device="cuda")
    ref = torch.eye(2, dtype=torch.float64, device="cuda")

    c_new, info = solve_region_confusion_update_pg(
        empirical=empirical,
        ideal_probabilities=ideal,
        confusion_init=c0,
        reference_confusion=ref,
        lambda_confusion=0.1,
        loss="l2",
        gamma_c=0.2,
        confusion_prev=c0,
        step_size=0.1,
        max_iters=20,
        tol=1e-8,
    )

    assert c_new.device.type == "cuda"
    validate_confusion_matrix(c_new, num_outcomes=2)
    assert info.num_iters >= 1


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the confusion_solver module.
    """
    tests = [
        ("confusion gradient finite difference (L2)", _finite_difference_check_confusion_gradient_l2),
        ("single-region fixed point", _self_test_region_confusion_solver_fixed_point),
        ("batch update shapes", _self_test_batch_update_shapes),
        ("truth fixed-point identity case", _self_test_truth_fixed_point_identity_case),
        ("gpu smoke", _self_test_gpu_smoke),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All confusion_solver self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
