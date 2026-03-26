from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Union

import torch

from alternating_solver import AlternatingSolverResult, solve_alternating
from config import (
    ADMMConfig,
    ExperimentConfig,
    LossConfig,
    RegionConfig,
    RuntimeConfig,
    SimulationConfig,
    build_pairwise_chain_regions,
    build_sliding_window_regions,
    make_default_experiment_config,
)
from metrics import summarize_history, summarize_solution
from noise import build_all_reference_confusions
from regions import RegionGraph
from simulator import SimulationResult, simulate_experiment


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


def _ensure_nonempty_string(value: str, name: str) -> str:
    value = str(value).strip()
    if len(value) == 0:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _normalize_qubits_per_site(
    qubits_per_site: Tuple[int, ...],
    num_sites: int,
) -> Tuple[int, ...]:
    if len(qubits_per_site) != int(num_sites):
        raise ValueError(
            f"qubits_per_site must have length {num_sites}, got {len(qubits_per_site)}."
        )
    out = tuple(int(q) for q in qubits_per_site)
    if any(q <= 0 for q in out):
        raise ValueError(f"All entries in qubits_per_site must be positive. Got {out}.")
    return out


def _make_base_experiment_config(
    *,
    qubits_per_site: Tuple[int, ...],
    regions: Tuple[RegionConfig, ...],
    loss_name: str = "nll",
    seed: int = 12345,
    experiment_name: str = "experiment",
) -> ExperimentConfig:
    """
    Build a standard ExperimentConfig with the project-wide torch/GPU runtime defaults.
    """
    return ExperimentConfig(
        qubits_per_site=qubits_per_site,
        regions=regions,
        loss=LossConfig(name=loss_name, prob_floor=1e-12),
        admm=ADMMConfig(
            beta=1.0,
            gamma_rho=1.0,
            gamma_c=1.0,
            lambda_confusion=1e-2,
            outer_max_iters=30,
            inner_max_iters=50,
            outer_tol=1e-6,
            inner_primal_tol=1e-6,
            inner_dual_tol=1e-6,
            state_step_size=0.1,
            confusion_step_size=0.1,
            state_gd_max_iters=100,
            confusion_gd_max_iters=100,
            state_gd_tol=1e-8,
            confusion_gd_tol=1e-8,
            verbose=True,
            store_history=True,
            print_every=1,
        ),
        simulation=SimulationConfig(
            seed=int(seed),
            use_shot_noise=True,
            state_rank=None,
            enforce_physical_truth=True,
        ),
        runtime=RuntimeConfig(
            device="auto",
            real_dtype="float64",
            complex_dtype="complex128",
            fallback_to_cpu=True,
            deterministic=False,
            float32_matmul_precision="high",
            num_threads=None,
        ),
        experiment_name=experiment_name,
    )


# ============================================================
# Experiment builders
# ============================================================

def make_pairwise_chain_experiment(
    *,
    num_sites: int,
    qubits_per_site: Tuple[int, ...],
    shots: int = 2000,
    povm_type: str = "random_ic",
    povm_num_outcomes: Optional[int] = None,
    true_state_model: str = "random_mixed",
    init_state_method: str = "maximally_mixed",
    true_confusion_model: str = "identity",
    init_confusion_method: str = "identity",
    confusion_strength: float = 0.05,
    loss_name: str = "nll",
    seed: int = 12345,
    experiment_name: str = "pairwise_chain_experiment",
) -> ExperimentConfig:
    """
    Build a pairwise-chain experiment with regions:
        (0,1), (1,2), ..., (n-2,n-1)
    """
    num_sites = _ensure_positive_int(num_sites, "num_sites")
    shots = _ensure_positive_int(shots, "shots")
    qubits_per_site = _normalize_qubits_per_site(qubits_per_site, num_sites)

    regions = build_pairwise_chain_regions(
        num_sites=num_sites,
        shots=shots,
        povm_type=povm_type,
        povm_num_outcomes=povm_num_outcomes,
        true_state_model=true_state_model,
        init_state_method=init_state_method,
        true_confusion_model=true_confusion_model,
        init_confusion_method=init_confusion_method,
        confusion_strength=float(confusion_strength),
        name_prefix="R",
    )

    return _make_base_experiment_config(
        qubits_per_site=qubits_per_site,
        regions=regions,
        loss_name=loss_name,
        seed=seed,
        experiment_name=experiment_name,
    )


def make_sliding_window_experiment(
    *,
    num_sites: int,
    window_size: int,
    qubits_per_site: Tuple[int, ...],
    shots: int = 2000,
    povm_type: str = "random_ic",
    povm_num_outcomes: Optional[int] = None,
    true_state_model: str = "random_mixed",
    init_state_method: str = "maximally_mixed",
    true_confusion_model: str = "identity",
    init_confusion_method: str = "identity",
    confusion_strength: float = 0.05,
    loss_name: str = "nll",
    seed: int = 12345,
    experiment_name: str = "sliding_window_experiment",
) -> ExperimentConfig:
    """
    Build a sliding-window experiment with regions:
        (0,...,w-1), (1,...,w), ..., (n-w,...,n-1)
    """
    num_sites = _ensure_positive_int(num_sites, "num_sites")
    window_size = _ensure_positive_int(window_size, "window_size")
    shots = _ensure_positive_int(shots, "shots")
    qubits_per_site = _normalize_qubits_per_site(qubits_per_site, num_sites)

    regions = build_sliding_window_regions(
        num_sites=num_sites,
        window_size=window_size,
        shots=shots,
        povm_type=povm_type,
        povm_num_outcomes=povm_num_outcomes,
        true_state_model=true_state_model,
        init_state_method=init_state_method,
        true_confusion_model=true_confusion_model,
        init_confusion_method=init_confusion_method,
        confusion_strength=float(confusion_strength),
        name_prefix="R",
    )

    return _make_base_experiment_config(
        qubits_per_site=qubits_per_site,
        regions=regions,
        loss_name=loss_name,
        seed=seed,
        experiment_name=experiment_name,
    )


def make_single_qubit_local_experiment(
    *,
    num_sites: int,
    shots: int = 1000,
    povm_type: str = "pauli6_single_qubit",
    true_state_model: str = "random_mixed",
    init_state_method: str = "maximally_mixed",
    true_confusion_model: str = "identity",
    init_confusion_method: str = "identity",
    confusion_strength: float = 0.05,
    loss_name: str = "nll",
    seed: int = 12345,
    experiment_name: str = "single_qubit_local_experiment",
) -> ExperimentConfig:
    """
    Build a local single-qubit experiment with one 1-qubit region per site.
    """
    num_sites = _ensure_positive_int(num_sites, "num_sites")
    shots = _ensure_positive_int(shots, "shots")
    qubits_per_site = tuple(1 for _ in range(num_sites))

    regions = build_sliding_window_regions(
        num_sites=num_sites,
        window_size=1,
        shots=shots,
        povm_type=povm_type,
        povm_num_outcomes=6 if povm_type == "pauli6_single_qubit" else None,
        true_state_model=true_state_model,
        init_state_method=init_state_method,
        true_confusion_model=true_confusion_model,
        init_confusion_method=init_confusion_method,
        confusion_strength=float(confusion_strength),
        name_prefix="R",
    )

    return _make_base_experiment_config(
        qubits_per_site=qubits_per_site,
        regions=regions,
        loss_name=loss_name,
        seed=seed,
        experiment_name=experiment_name,
    )


def make_fast_debug_experiment() -> ExperimentConfig:
    """
    Small fast-turnaround preset used for smoke tests and debugging.
    """
    cfg = make_default_experiment_config()
    cfg.experiment_name = "fast_debug_experiment"
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False
    cfg.admm.outer_max_iters = 2
    cfg.admm.inner_max_iters = 5
    cfg.admm.state_gd_max_iters = 20
    cfg.admm.confusion_gd_max_iters = 20
    cfg.admm.verbose = False
    return cfg


def make_paper_pairwise_chain_baseline() -> ExperimentConfig:
    """
    Stabilized baseline used for current paper-style experiments.

    Geometry
    --------
    - 4 sites
    - 1 qubit per site
    - pairwise chain regions: (0,1), (1,2), (2,3)

    Measurement / truth
    -------------------
    - random informationally complete regional POVMs with 16 outcomes
    - random mixed true regional states
    - noisy-identity true confusion matrices
    - identity initialization for confusion matrices
    - shot noise enabled
    - NLL loss

    Tuned solver settings
    ---------------------
    These settings were selected after the diagnostic tuning sequence and are
    the recommended current baseline on this geometry.
    """
    cfg = make_pairwise_chain_experiment(
        num_sites=4,
        qubits_per_site=(1, 1, 1, 1),
        shots=1500,
        povm_type="random_ic",
        povm_num_outcomes=16,
        true_state_model="random_mixed",
        init_state_method="maximally_mixed",
        true_confusion_model="noisy_identity",
        init_confusion_method="identity",
        confusion_strength=0.05,
        loss_name="nll",
        seed=12347,
        experiment_name="paper_pairwise_chain_baseline",
    )

    cfg.simulation.use_shot_noise = True

    cfg.admm.beta = 4.0
    cfg.admm.gamma_rho = 1.0
    cfg.admm.gamma_c = 40.0
    cfg.admm.lambda_confusion = 0.5

    cfg.admm.outer_max_iters = 8
    cfg.admm.inner_max_iters = 60

    cfg.admm.state_gd_max_iters = 60
    cfg.admm.confusion_gd_max_iters = 400

    cfg.admm.confusion_step_size = 0.03
    cfg.admm.confusion_gd_tol = 1e-6

    cfg.admm.verbose = False
    cfg.admm.print_every = 1

    return cfg


def make_named_experiment(name: str) -> ExperimentConfig:
    """
    Build one of the named preset experiments.

    Supported names
    ---------------
    - "default"
    - "fast_debug"
    - "pairwise_chain_small"
    - "sliding_window_small"
    - "single_qubit_local"
    - "paper_pairwise_chain_baseline"
    """
    name = _ensure_nonempty_string(name, "name").lower()

    if name == "default":
        return make_default_experiment_config()

    if name == "fast_debug":
        return make_fast_debug_experiment()

    if name == "pairwise_chain_small":
        return make_pairwise_chain_experiment(
            num_sites=4,
            qubits_per_site=(1, 1, 1, 1),
            shots=800,
            povm_type="random_ic",
            povm_num_outcomes=16,
            true_state_model="random_mixed",
            init_state_method="maximally_mixed",
            true_confusion_model="noisy_identity",
            init_confusion_method="identity",
            confusion_strength=0.03,
            loss_name="nll",
            seed=12345,
            experiment_name="pairwise_chain_small",
        )

    if name == "sliding_window_small":
        return make_sliding_window_experiment(
            num_sites=5,
            window_size=3,
            qubits_per_site=(1, 1, 1, 1, 1),
            shots=800,
            povm_type="random_ic",
            povm_num_outcomes=64,  # 3 qubits -> dim=8 -> dim^2=64
            true_state_model="random_mixed",
            init_state_method="maximally_mixed",
            true_confusion_model="noisy_identity",
            init_confusion_method="identity",
            confusion_strength=0.03,
            loss_name="nll",
            seed=12345,
            experiment_name="sliding_window_small",
        )

    if name == "single_qubit_local":
        return make_single_qubit_local_experiment(
            num_sites=4,
            shots=1000,
            povm_type="pauli6_single_qubit",
            true_state_model="random_mixed",
            init_state_method="maximally_mixed",
            true_confusion_model="noisy_identity",
            init_confusion_method="identity",
            confusion_strength=0.03,
            loss_name="nll",
            seed=12345,
            experiment_name="single_qubit_local",
        )

    if name == "paper_pairwise_chain_baseline":
        return make_paper_pairwise_chain_baseline()

    supported = (
        "default",
        "fast_debug",
        "pairwise_chain_small",
        "sliding_window_small",
        "single_qubit_local",
        "paper_pairwise_chain_baseline",
    )
    raise ValueError(f"Unknown experiment name '{name}'. Supported names: {supported}.")


def list_available_experiments() -> Tuple[str, ...]:
    """
    Return the available named preset experiments.
    """
    return (
        "default",
        "fast_debug",
        "pairwise_chain_small",
        "sliding_window_small",
        "single_qubit_local",
        "paper_pairwise_chain_baseline",
    )


# ============================================================
# End-to-end runner
# ============================================================

@dataclass
class ExperimentRunResult:
    """
    Bundle for a full experiment execution.

    Attributes
    ----------
    config :
        Experiment configuration used.
    simulation :
        Synthetic-data simulation bundle.
    solver_result :
        Outer alternating-solver result.
    summary :
        Final metrics summary.
    history_summary :
        Summary statistics for solver history arrays.
    """
    config: ExperimentConfig
    simulation: SimulationResult
    solver_result: AlternatingSolverResult
    summary: Dict[str, object]
    history_summary: Dict[str, Dict[str, float]]

    def validate(self) -> None:
        """
        Validate the combined run result.
        """
        graph = RegionGraph(self.config)
        self.simulation.validate()
        self.solver_result.validate(self.config, graph)

    def pretty_print(self) -> None:
        """
        Print a compact end-to-end run summary.
        """
        print("=" * 72)
        print("ExperimentRunResult")
        print("-" * 72)
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Solver converged: {self.solver_result.converged}")
        print(f"Outer iterations: {self.solver_result.num_outer_iterations}")
        print(f"Final objective: {self.solver_result.final_objective:.6e}")
        print(
            f"Final state overlap residual: "
            f"{self.solver_result.final_state_max_overlap_residual:.6e}"
        )
        if "fit_objective" in self.summary:
            print(f"Fit objective: {float(self.summary['fit_objective']):.6e}")
        if "regularized_objective" in self.summary:
            print(
                f"Regularized objective: "
                f"{float(self.summary['regularized_objective']):.6e}"
            )
        if "state_error" in self.summary:
            print(
                f"Mean state error: "
                f"{float(self.summary['state_error']['aggregate']['mean']):.6e}"
            )
        if "confusion_error" in self.summary:
            print(
                f"Mean confusion error: "
                f"{float(self.summary['confusion_error']['aggregate']['mean']):.6e}"
            )
        print("=" * 72)


def run_configured_experiment(
    cfg: ExperimentConfig,
    *,
    truth_mode: str = "global_consistent",
    site_model_override: Optional[str] = None,
    initial_region_states: Optional[Mapping[str, torch.Tensor]] = None,
    initial_region_confusions: Optional[Mapping[str, torch.Tensor]] = None,
    loss: Optional[Union[str, LossConfig]] = None,
    prob_floor: Optional[float] = None,
    verbose: Optional[bool] = None,
) -> ExperimentRunResult:
    """
    Run a complete configured experiment:
        simulate -> solve -> summarize
    """
    simulation = simulate_experiment(
        cfg=cfg,
        truth_mode=truth_mode,
        site_model_override=site_model_override,
    )

    reference_confusions = build_all_reference_confusions(cfg)

    solver_result = solve_alternating(
        cfg=cfg,
        empirical_probabilities=simulation.empirical_probabilities,
        region_povms=simulation.region_povms,
        initial_region_states=initial_region_states,
        initial_region_confusions=initial_region_confusions,
        reference_confusions=reference_confusions,
        loss=loss,
        region_shots=simulation.region_shots,
        prob_floor=prob_floor,
        verbose=verbose,
    )

    summary = summarize_solution(
        cfg=cfg,
        empirical_probabilities=simulation.empirical_probabilities,
        region_states=solver_result.region_states,
        region_povms=simulation.region_povms,
        region_confusions=solver_result.region_confusions,
        reference_confusions=reference_confusions,
        true_region_states=simulation.region_states,
        true_region_confusions=simulation.region_confusions,
        loss=loss,
        region_shots=simulation.region_shots,
        prob_floor=prob_floor,
    )

    history_summary = summarize_history(solver_result.history)

    result = ExperimentRunResult(
        config=cfg,
        simulation=simulation,
        solver_result=solver_result,
        summary=summary,
        history_summary=history_summary,
    )
    result.validate()
    return result


def run_named_experiment(
    name: str,
    *,
    truth_mode: str = "global_consistent",
    site_model_override: Optional[str] = None,
    loss: Optional[Union[str, LossConfig]] = None,
    prob_floor: Optional[float] = None,
    verbose: Optional[bool] = None,
) -> ExperimentRunResult:
    """
    Convenience wrapper:
        cfg = make_named_experiment(name)
        run_configured_experiment(cfg, ...)
    """
    cfg = make_named_experiment(name)
    return run_configured_experiment(
        cfg,
        truth_mode=truth_mode,
        site_model_override=site_model_override,
        loss=loss,
        prob_floor=prob_floor,
        verbose=verbose,
    )


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_pairwise_builder() -> None:
    cfg = make_pairwise_chain_experiment(
        num_sites=4,
        qubits_per_site=(1, 1, 1, 1),
        shots=500,
        povm_type="random_ic",
        povm_num_outcomes=16,
        experiment_name="pairwise_test",
    )
    graph = RegionGraph(cfg)

    assert cfg.num_sites == 4
    assert cfg.num_regions == 3
    assert graph.overlap_pairs() == ((0, 1), (1, 2))


def _self_test_sliding_builder() -> None:
    cfg = make_sliding_window_experiment(
        num_sites=5,
        window_size=3,
        qubits_per_site=(1, 1, 1, 1, 1),
        shots=500,
        povm_type="random_ic",
        povm_num_outcomes=64,
        experiment_name="sliding_test",
    )
    graph = RegionGraph(cfg)

    assert cfg.num_regions == 3
    # Overlap is any nonempty intersection, so R0 and R2 also overlap at site 2.
    assert graph.overlap_pairs() == ((0, 1), (0, 2), (1, 2))


def _self_test_named_builder() -> None:
    cfg = make_named_experiment("fast_debug")
    assert cfg.experiment_name == "fast_debug_experiment"
    assert cfg.num_regions >= 1


def _self_test_paper_baseline_builder() -> None:
    cfg = make_named_experiment("paper_pairwise_chain_baseline")
    graph = RegionGraph(cfg)

    assert cfg.experiment_name == "paper_pairwise_chain_baseline"
    assert cfg.num_sites == 4
    assert cfg.num_regions == 3
    assert graph.overlap_pairs() == ((0, 1), (1, 2))
    assert cfg.loss.name == "nll"
    assert cfg.simulation.use_shot_noise is True
    assert cfg.admm.beta == 4.0
    assert cfg.admm.gamma_c == 40.0
    assert cfg.admm.lambda_confusion == 0.5
    assert cfg.admm.outer_max_iters == 8
    assert cfg.admm.inner_max_iters == 60
    assert cfg.admm.confusion_step_size == 0.03
    assert cfg.admm.confusion_gd_max_iters == 400
    assert cfg.admm.confusion_gd_tol == 1e-6


def _self_test_end_to_end_run() -> None:
    cfg = make_fast_debug_experiment()
    result = run_configured_experiment(cfg, truth_mode="global_consistent", verbose=False)

    result.validate()
    assert result.solver_result.num_outer_iterations >= 1
    assert "fit_objective" in result.summary
    assert "objective" in result.history_summary


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the experiments module.
    """
    tests = [
        ("pairwise builder", _self_test_pairwise_builder),
        ("sliding-window builder", _self_test_sliding_builder),
        ("named preset builder", _self_test_named_builder),
        ("paper baseline builder", _self_test_paper_baseline_builder),
        ("end-to-end experiment run", _self_test_end_to_end_run),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All experiments self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
