from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch

from config import ExperimentConfig, LossConfig
from confusion_solver import ConfusionUpdateResult, update_all_confusions
from core_ops import frobenius_norm
from measurements import POVM, validate_region_povm_collection
from noise import (
    build_all_initial_confusions,
    build_all_reference_confusions,
    validate_region_confusion_collection,
)
from objectives import (
    build_region_shot_dict,
    relative_change_dict,
    total_regularized_objective,
)
from regions import RegionGraph
from state_admm import (
    StateADMMResult,
    initialize_eta_from_region_states,
    initialize_zero_duals,
    solve_state_subproblem_admm,
    validate_dual_collection,
    validate_eta_collection,
)
from states import (
    initialize_all_region_states,
    validate_region_state_collection,
)


# ============================================================
# Small helpers
# ============================================================

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


def _as_torch_tensor(x, dtype=None, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(
            dtype=x.dtype if dtype is None else dtype,
            device=x.device if device is None else device,
        )
    return torch.as_tensor(x, dtype=dtype, device=device)


def _copy_matrix_dict(d: Mapping) -> Dict:
    return {k: _as_torch_tensor(v).clone() for k, v in d.items()}


def _relative_scalar_change(current: float, previous: float, eps: float = 1e-12) -> float:
    return float(abs(float(current) - float(previous)) / max(abs(float(previous)), eps))


# ============================================================
# Initialization helpers
# ============================================================

def initialize_alternating_iterates(
    cfg: ExperimentConfig,
    *,
    initial_region_states: Optional[Mapping[str, torch.Tensor]] = None,
    initial_region_confusions: Optional[Mapping[str, torch.Tensor]] = None,
    reference_confusions: Optional[Mapping[str, torch.Tensor]] = None,
    graph: Optional[RegionGraph] = None,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[Tuple[str, str], torch.Tensor],
    Dict[Tuple[str, str], torch.Tensor],
]:
    """
    Initialize the outer alternating-solver iterates.

    Returns
    -------
    tuple
        (region_states, region_confusions, reference_confusions, eta, duals)
    """
    graph = RegionGraph(cfg) if graph is None else graph
    cfg.apply_runtime()

    if initial_region_states is None:
        rng = cfg.make_torch_generator()
        region_states = initialize_all_region_states(cfg, rng=rng)
    else:
        region_states = _copy_matrix_dict(initial_region_states)
    validate_region_state_collection(cfg, region_states, check_overlap_consistency=False)

    if initial_region_confusions is None:
        rng = cfg.make_torch_generator()
        region_confusions = build_all_initial_confusions(cfg, rng=rng)
    else:
        region_confusions = _copy_matrix_dict(initial_region_confusions)
    validate_region_confusion_collection(cfg, region_confusions)

    if reference_confusions is None:
        reference_confusions_out = build_all_reference_confusions(cfg)
    else:
        reference_confusions_out = _copy_matrix_dict(reference_confusions)
    validate_region_confusion_collection(cfg, reference_confusions_out)

    eta = initialize_eta_from_region_states(graph, region_states, average_pair_reductions=True)
    duals = initialize_zero_duals(
        graph,
        dtype=cfg.torch_complex_dtype,
        device=cfg.device,
    )

    validate_eta_collection(graph, eta)
    validate_dual_collection(graph, duals)

    return region_states, region_confusions, reference_confusions_out, eta, duals


# ============================================================
# Result object
# ============================================================

@dataclass
class AlternatingSolverResult:
    region_states: Dict[str, torch.Tensor]
    region_confusions: Dict[str, torch.Tensor]
    reference_confusions: Dict[str, torch.Tensor]
    eta: Dict[Tuple[str, str], torch.Tensor]
    duals: Dict[Tuple[str, str], torch.Tensor]

    converged: bool
    num_outer_iterations: int

    initial_objective: float
    final_objective: float

    final_state_primal_residual: float
    final_state_dual_residual: float
    final_state_max_overlap_residual: float
    final_confusion_average_pg_iters: float

    history: Dict[str, List[float]]

    def validate(self, cfg: ExperimentConfig, graph: Optional[RegionGraph] = None) -> None:
        """
        Validate the outer-solver result object.
        """
        graph = RegionGraph(cfg) if graph is None else graph

        validate_region_state_collection(cfg, self.region_states, check_overlap_consistency=False)
        validate_region_confusion_collection(cfg, self.region_confusions)
        validate_region_confusion_collection(cfg, self.reference_confusions)
        validate_eta_collection(graph, self.eta)
        validate_dual_collection(graph, self.duals)

    def pretty_print(self) -> None:
        print("=" * 72)
        print("AlternatingSolverResult")
        print("-" * 72)
        print(f"Converged: {self.converged}")
        print(f"Outer iterations: {self.num_outer_iterations}")
        print(f"Initial objective: {self.initial_objective:.6e}")
        print(f"Final objective: {self.final_objective:.6e}")
        print(f"Final state primal residual: {self.final_state_primal_residual:.6e}")
        print(f"Final state dual residual: {self.final_state_dual_residual:.6e}")
        print(f"Final max overlap residual: {self.final_state_max_overlap_residual:.6e}")
        print(f"Final confusion average PG iters: {self.final_confusion_average_pg_iters:.2f}")
        print("=" * 72)


# ============================================================
# Main outer solver
# ============================================================

def solve_alternating(
    cfg: ExperimentConfig,
    empirical_probabilities: Mapping[str, torch.Tensor],
    region_povms: Mapping[str, POVM],
    *,
    initial_region_states: Optional[Mapping[str, torch.Tensor]] = None,
    initial_region_confusions: Optional[Mapping[str, torch.Tensor]] = None,
    reference_confusions: Optional[Mapping[str, torch.Tensor]] = None,
    loss: Optional[Union[str, LossConfig]] = None,
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
    outer_max_iters: Optional[int] = None,
    outer_tol: Optional[float] = None,
    beta: Optional[float] = None,
    gamma_rho: Optional[float] = None,
    gamma_c: Optional[float] = None,
    lambda_confusion: Optional[float] = None,
    inner_max_iters: Optional[int] = None,
    inner_primal_tol: Optional[float] = None,
    inner_dual_tol: Optional[float] = None,
    state_step_size: Optional[float] = None,
    state_gd_max_iters: Optional[int] = None,
    state_gd_tol: Optional[float] = None,
    confusion_step_size: Optional[float] = None,
    confusion_gd_max_iters: Optional[int] = None,
    confusion_gd_tol: Optional[float] = None,
    verbose: Optional[bool] = None,
    print_every: Optional[int] = None,
) -> AlternatingSolverResult:
    """
    Run the full proximal alternating scheme.

    Outer loop
    ----------
    1) Solve the state block by inner ADMM for fixed confusion matrices.
    2) Update all confusion matrices locally for fixed states.
    3) Check outer convergence.

    Returns
    -------
    AlternatingSolverResult
        Final outer-solver output and diagnostics.
    """
    cfg.apply_runtime()
    graph = RegionGraph(cfg)
    loss = _resolve_loss(loss, cfg)
    prob_floor = _resolve_prob_floor(loss, prob_floor)

    outer_max_iters = cfg.admm.outer_max_iters if outer_max_iters is None else outer_max_iters
    outer_tol = cfg.admm.outer_tol if outer_tol is None else outer_tol
    beta = cfg.admm.beta if beta is None else beta
    gamma_rho = cfg.admm.gamma_rho if gamma_rho is None else gamma_rho
    gamma_c = cfg.admm.gamma_c if gamma_c is None else gamma_c
    lambda_confusion = cfg.admm.lambda_confusion if lambda_confusion is None else lambda_confusion

    inner_max_iters = cfg.admm.inner_max_iters if inner_max_iters is None else inner_max_iters
    inner_primal_tol = cfg.admm.inner_primal_tol if inner_primal_tol is None else inner_primal_tol
    inner_dual_tol = cfg.admm.inner_dual_tol if inner_dual_tol is None else inner_dual_tol

    state_step_size = cfg.admm.state_step_size if state_step_size is None else state_step_size
    state_gd_max_iters = cfg.admm.state_gd_max_iters if state_gd_max_iters is None else state_gd_max_iters
    state_gd_tol = cfg.admm.state_gd_tol if state_gd_tol is None else state_gd_tol

    confusion_step_size = cfg.admm.confusion_step_size if confusion_step_size is None else confusion_step_size
    confusion_gd_max_iters = (
        cfg.admm.confusion_gd_max_iters if confusion_gd_max_iters is None else confusion_gd_max_iters
    )
    confusion_gd_tol = cfg.admm.confusion_gd_tol if confusion_gd_tol is None else confusion_gd_tol

    verbose = cfg.admm.verbose if verbose is None else bool(verbose)
    print_every = cfg.admm.print_every if print_every is None else print_every

    outer_max_iters = _ensure_positive_int(outer_max_iters, "outer_max_iters")
    outer_tol = _ensure_positive_float(outer_tol, "outer_tol")
    beta = _ensure_positive_float(beta, "beta")
    gamma_rho = _ensure_positive_float(gamma_rho, "gamma_rho")
    gamma_c = _ensure_positive_float(gamma_c, "gamma_c")
    lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")

    inner_max_iters = _ensure_positive_int(inner_max_iters, "inner_max_iters")
    inner_primal_tol = _ensure_positive_float(inner_primal_tol, "inner_primal_tol")
    inner_dual_tol = _ensure_positive_float(inner_dual_tol, "inner_dual_tol")

    state_step_size = _ensure_positive_float(state_step_size, "state_step_size")
    state_gd_max_iters = _ensure_positive_int(state_gd_max_iters, "state_gd_max_iters")
    state_gd_tol = _ensure_positive_float(state_gd_tol, "state_gd_tol")

    confusion_step_size = _ensure_positive_float(confusion_step_size, "confusion_step_size")
    confusion_gd_max_iters = _ensure_positive_int(confusion_gd_max_iters, "confusion_gd_max_iters")
    confusion_gd_tol = _ensure_positive_float(confusion_gd_tol, "confusion_gd_tol")

    print_every = _ensure_positive_int(print_every, "print_every")

    expected_names = {region.name for region in cfg.regions}
    if set(empirical_probabilities.keys()) != expected_names:
        raise ValueError("empirical_probabilities keys must match cfg.regions.")
    if set(region_povms.keys()) != expected_names:
        raise ValueError("region_povms keys must match cfg.regions.")

    validate_region_povm_collection(cfg, dict(region_povms))

    if region_shots is None:
        region_shots = build_region_shot_dict(cfg)
    elif set(region_shots.keys()) != expected_names:
        raise ValueError("region_shots keys must match cfg.regions.")

    (
        region_states_curr,
        region_confusions_curr,
        reference_confusions_curr,
        eta_curr,
        duals_curr,
    ) = initialize_alternating_iterates(
        cfg,
        initial_region_states=initial_region_states,
        initial_region_confusions=initial_region_confusions,
        reference_confusions=reference_confusions,
        graph=graph,
    )

    initial_objective = total_regularized_objective(
        empirical_probabilities=empirical_probabilities,
        region_states=region_states_curr,
        region_povms=region_povms,
        region_confusions=region_confusions_curr,
        reference_confusions=reference_confusions_curr,
        lambda_confusion=lambda_confusion,
        loss=loss,
        region_shots=region_shots,
        prob_floor=prob_floor,
    )
    current_objective = float(initial_objective)

    history: Dict[str, List[float]] = {
        "objective": [],
        "state_change": [],
        "confusion_change": [],
        "objective_relative_change": [],
        "state_primal_residual": [],
        "state_dual_residual": [],
        "state_max_overlap_residual": [],
        "confusion_average_pg_iters": [],
    }

    converged = False
    num_outer_iterations = 0

    final_state_primal = float("nan")
    final_state_dual = float("nan")
    final_state_max_overlap = float("nan")
    final_confusion_avg_pg_iters = float("nan")

    for outer_it in range(1, outer_max_iters + 1):
        num_outer_iterations = outer_it

        states_prev = _copy_matrix_dict(region_states_curr)
        confusions_prev = _copy_matrix_dict(region_confusions_curr)
        objective_prev = float(current_objective)

        # ----------------------------------------------------
        # 1) State block update via inner ADMM
        # ----------------------------------------------------
        state_result: StateADMMResult = solve_state_subproblem_admm(
            cfg=cfg,
            empirical_probabilities=empirical_probabilities,
            region_povms=region_povms,
            fixed_confusions=confusions_prev,
            region_states_outer_prev=states_prev,
            eta_init=eta_curr,
            duals_init=duals_curr,
            graph=graph,
            loss=loss,
            region_shots=region_shots,
            prob_floor=prob_floor,
            beta=beta,
            gamma_rho=gamma_rho,
            inner_max_iters=inner_max_iters,
            inner_primal_tol=inner_primal_tol,
            inner_dual_tol=inner_dual_tol,
            state_step_size=state_step_size,
            state_gd_max_iters=state_gd_max_iters,
            state_gd_tol=state_gd_tol,
            verbose=False,
        )

        region_states_next = state_result.region_states
        eta_next = state_result.eta
        duals_next = state_result.duals

        # ----------------------------------------------------
        # 2) Confusion block update
        # ----------------------------------------------------
        confusion_result: ConfusionUpdateResult = update_all_confusions(
            cfg=cfg,
            empirical_probabilities=empirical_probabilities,
            region_states_fixed=region_states_next,
            region_povms=region_povms,
            confusion_prev=confusions_prev,
            reference_confusions=reference_confusions_curr,
            loss=loss,
            region_shots=region_shots,
            prob_floor=prob_floor,
            lambda_confusion=lambda_confusion,
            gamma_c=gamma_c,
            step_size=confusion_step_size,
            max_iters=confusion_gd_max_iters,
            tol=confusion_gd_tol,
            verbose=False,
        )

        region_confusions_next = confusion_result.region_confusions

        # ----------------------------------------------------
        # 3) Diagnostics
        # ----------------------------------------------------
        current_objective = total_regularized_objective(
            empirical_probabilities=empirical_probabilities,
            region_states=region_states_next,
            region_povms=region_povms,
            region_confusions=region_confusions_next,
            reference_confusions=reference_confusions_curr,
            lambda_confusion=lambda_confusion,
            loss=loss,
            region_shots=region_shots,
            prob_floor=prob_floor,
        )

        state_change = relative_change_dict(region_states_next, states_prev)
        confusion_change = relative_change_dict(region_confusions_next, confusions_prev)
        objective_rel_change = _relative_scalar_change(current_objective, objective_prev)

        final_state_primal = float(state_result.final_primal_residual)
        final_state_dual = float(state_result.final_dual_residual)
        final_state_max_overlap = float(state_result.final_max_overlap_residual)
        final_confusion_avg_pg_iters = float(confusion_result.average_pg_iters)

        history["objective"].append(float(current_objective))
        history["state_change"].append(float(state_change))
        history["confusion_change"].append(float(confusion_change))
        history["objective_relative_change"].append(float(objective_rel_change))
        history["state_primal_residual"].append(final_state_primal)
        history["state_dual_residual"].append(final_state_dual)
        history["state_max_overlap_residual"].append(final_state_max_overlap)
        history["confusion_average_pg_iters"].append(final_confusion_avg_pg_iters)

        if verbose and (outer_it == 1 or outer_it % print_every == 0):
            print(
                f"[alternating_solver] outer={outer_it:03d} "
                f"obj={current_objective:.6e} "
                f"dobj={objective_rel_change:.6e} "
                f"dstate={state_change:.6e} "
                f"dconf={confusion_change:.6e} "
                f"sprimal={final_state_primal:.6e} "
                f"sdual={final_state_dual:.6e} "
                f"avgCpg={final_confusion_avg_pg_iters:.2f}"
            )

        # Update current iterates
        region_states_curr = region_states_next
        region_confusions_curr = region_confusions_next
        eta_curr = eta_next
        duals_curr = duals_next

        # Outer stopping criterion
        if (
            state_change <= outer_tol
            and confusion_change <= outer_tol
            and objective_rel_change <= outer_tol
        ):
            converged = True
            break

    result = AlternatingSolverResult(
        region_states=region_states_curr,
        region_confusions=region_confusions_curr,
        reference_confusions=reference_confusions_curr,
        eta=eta_curr,
        duals=duals_curr,
        converged=converged,
        num_outer_iterations=num_outer_iterations,
        initial_objective=float(initial_objective),
        final_objective=float(current_objective),
        final_state_primal_residual=float(final_state_primal),
        final_state_dual_residual=float(final_state_dual),
        final_state_max_overlap_residual=float(final_state_max_overlap),
        final_confusion_average_pg_iters=float(final_confusion_avg_pg_iters),
        history=history,
    )
    result.validate(cfg, graph)
    return result


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_initialization_helper() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    graph = RegionGraph(cfg)

    states0, confs0, refs, eta0, duals0 = initialize_alternating_iterates(cfg, graph=graph)

    validate_region_state_collection(cfg, states0, check_overlap_consistency=False)
    validate_region_confusion_collection(cfg, confs0)
    validate_region_confusion_collection(cfg, refs)
    validate_eta_collection(graph, eta0)
    validate_dual_collection(graph, duals0)


def _self_test_fixed_point_identity_case() -> None:
    from config import make_default_experiment_config
    from simulator import simulate_experiment

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    for region in cfg.regions:
        region.true_confusion_model = "identity"
        region.init_confusion_method = "identity"

    sim = simulate_experiment(cfg, truth_mode="global_consistent")

    result = solve_alternating(
        cfg=cfg,
        empirical_probabilities=sim.empirical_probabilities,
        region_povms=sim.region_povms,
        initial_region_states=sim.region_states,
        initial_region_confusions=sim.region_confusions,
        reference_confusions=build_all_reference_confusions(cfg),
        loss="l2",
        region_shots=sim.region_shots,
        outer_max_iters=3,
        outer_tol=1e-10,
        inner_max_iters=5,
        inner_primal_tol=1e-10,
        inner_dual_tol=1e-10,
        state_step_size=0.1,
        state_gd_max_iters=50,
        state_gd_tol=1e-10,
        confusion_step_size=0.1,
        confusion_gd_max_iters=50,
        confusion_gd_tol=1e-10,
        verbose=False,
    )

    result.validate(cfg)
    assert result.final_objective <= 1e-8

    for region in cfg.regions:
        name = region.name
        assert frobenius_norm(result.region_states[name] - sim.region_states[name]) <= 1e-7
        assert frobenius_norm(result.region_confusions[name] - sim.region_confusions[name]) <= 1e-7


def _self_test_general_run() -> None:
    from config import make_default_experiment_config
    from simulator import simulate_experiment

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    sim = simulate_experiment(cfg, truth_mode="global_consistent")

    result = solve_alternating(
        cfg=cfg,
        empirical_probabilities=sim.empirical_probabilities,
        region_povms=sim.region_povms,
        loss="l2",
        region_shots=sim.region_shots,
        outer_max_iters=2,
        outer_tol=1e-8,
        inner_max_iters=5,
        inner_primal_tol=1e-8,
        inner_dual_tol=1e-8,
        state_step_size=0.1,
        state_gd_max_iters=20,
        state_gd_tol=1e-8,
        confusion_step_size=0.1,
        confusion_gd_max_iters=20,
        confusion_gd_tol=1e-8,
        verbose=False,
    )

    result.validate(cfg)
    assert result.num_outer_iterations >= 1
    assert len(result.history["objective"]) == result.num_outer_iterations
    assert set(result.region_states.keys()) == {region.name for region in cfg.regions}
    assert set(result.region_confusions.keys()) == {region.name for region in cfg.regions}

    for region in cfg.regions:
        assert tuple(result.region_states[region.name].shape) == (
            cfg.region_dimension(region),
            cfg.region_dimension(region),
        )


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the alternating_solver module.
    """
    tests = [
        ("initialization helper", _self_test_initialization_helper),
        ("fixed-point identity case", _self_test_fixed_point_identity_case),
        ("general alternating run", _self_test_general_run),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All alternating_solver self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
