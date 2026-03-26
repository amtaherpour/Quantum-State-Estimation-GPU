from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch

from config import ExperimentConfig
from measurements import POVM, build_all_region_povms, measurement_map, validate_region_povm_collection
from noise import (
    apply_confusion_matrix,
    build_all_true_confusions,
    validate_region_confusion_collection,
)
from states import (
    generate_consistent_regional_truth_from_global_product,
    generate_independent_regional_truth,
    validate_region_state_collection,
)


# ============================================================
# Small helpers
# ============================================================

def _coerce_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _coerce_real_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        return torch.float64
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}.")
    return dtype


def _real_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    if dtype in {torch.float32, torch.float64}:
        return dtype
    raise ValueError(f"Unsupported dtype: {dtype}.")


def _as_torch_tensor(
    x,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(
            dtype=x.dtype if dtype is None else dtype,
            device=x.device if device is None else device,
        )
    return torch.as_tensor(x, dtype=dtype, device=device)


def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _coerce_rng(
    rng: Optional[Union[int, torch.Generator]],
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Generator:
    device = _coerce_device(device)

    if isinstance(rng, torch.Generator):
        return rng
    gen = torch.Generator(device=device.type)
    if rng is None:
        gen.seed()
    else:
        gen.manual_seed(int(rng))
    return gen


def _normalize_probability_vector(p: torch.Tensor, name: str) -> torch.Tensor:
    p = _as_torch_tensor(p).reshape(-1)
    if p.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not p.dtype.is_floating_point:
        p = p.to(torch.float64)
    if torch.is_complex(p):
        raise ValueError(f"{name} must be real-valued.")
    if not torch.all(torch.isfinite(p)):
        raise ValueError(f"{name} contains non-finite values.")
    if torch.any(p < -1e-12):
        raise ValueError(f"{name} contains negative entries.")
    p = torch.clamp(p, min=0.0)
    s = torch.sum(p)
    if float(s.item()) <= 0.0:
        raise ValueError(f"{name} must have positive total mass.")
    return p / s


def _resolve_global_site_model(cfg: ExperimentConfig, site_model_override: Optional[str]) -> str:
    if site_model_override is not None:
        return str(site_model_override)

    models = {region.true_state_model for region in cfg.regions}
    if len(models) == 1:
        return next(iter(models))

    raise ValueError(
        "The regions do not share a common true_state_model, so a consistent global truth "
        "cannot be inferred automatically. Pass site_model_override explicitly."
    )


# ============================================================
# Data containers
# ============================================================

@dataclass
class SimulationResult:
    """
    Bundle containing a complete synthetic experiment.
    """
    config: ExperimentConfig
    global_state: Optional[torch.Tensor]
    site_states: Optional[Tuple[torch.Tensor, ...]]
    region_states: Dict[str, torch.Tensor]
    region_povms: Dict[str, POVM]
    region_confusions: Dict[str, torch.Tensor]
    ideal_probabilities: Dict[str, torch.Tensor]
    noisy_probabilities: Dict[str, torch.Tensor]
    counts: Dict[str, torch.Tensor]
    empirical_probabilities: Dict[str, torch.Tensor]
    region_shots: Dict[str, int]
    metadata: Dict[str, object]

    def validate(self) -> None:
        """
        Validate internal consistency of the simulation bundle.
        """
        validate_region_state_collection(
            self.config,
            self.region_states,
            check_overlap_consistency=self.global_state is not None,
        )
        validate_region_povm_collection(self.config, self.region_povms)
        validate_region_confusion_collection(self.config, self.region_confusions)

        expected_names = {region.name for region in self.config.regions}

        collections = {
            "region_povms": self.region_povms,
            "ideal_probabilities": self.ideal_probabilities,
            "noisy_probabilities": self.noisy_probabilities,
            "counts": self.counts,
            "empirical_probabilities": self.empirical_probabilities,
            "region_shots": self.region_shots,
        }

        for label, mapping in collections.items():
            names = set(mapping.keys())
            if names != expected_names:
                missing = expected_names - names
                extra = names - expected_names
                raise ValueError(
                    f"{label} keys do not match configured region names. "
                    f"Missing={sorted(missing)}, extra={sorted(extra)}."
                )

        for region in self.config.regions:
            name = region.name
            m = self.region_povms[name].num_outcomes
            shots = int(self.region_shots[name])

            ideal = self.ideal_probabilities[name]
            noisy = self.noisy_probabilities[name]
            empirical = self.empirical_probabilities[name]
            counts = self.counts[name]

            if tuple(ideal.shape) != (m,):
                raise ValueError(
                    f"ideal_probabilities['{name}'] has shape {tuple(ideal.shape)}, expected {(m,)}."
                )
            if tuple(noisy.shape) != (m,):
                raise ValueError(
                    f"noisy_probabilities['{name}'] has shape {tuple(noisy.shape)}, expected {(m,)}."
                )
            if tuple(empirical.shape) != (m,):
                raise ValueError(
                    f"empirical_probabilities['{name}'] has shape {tuple(empirical.shape)}, expected {(m,)}."
                )
            if tuple(counts.shape) != (m,):
                raise ValueError(
                    f"counts['{name}'] has shape {tuple(counts.shape)}, expected {(m,)}."
                )

            if int(torch.sum(counts).item()) != shots:
                raise ValueError(
                    f"counts['{name}'] sums to {int(torch.sum(counts).item())}, expected {shots}."
                )

            one_ideal = torch.tensor(1.0, dtype=ideal.dtype, device=ideal.device)
            one_noisy = torch.tensor(1.0, dtype=noisy.dtype, device=noisy.device)
            one_emp = torch.tensor(1.0, dtype=empirical.dtype, device=empirical.device)

            if not torch.isclose(torch.sum(ideal), one_ideal, atol=1e-10, rtol=1e-8):
                raise ValueError(f"ideal_probabilities['{name}'] does not sum to 1.")
            if not torch.isclose(torch.sum(noisy), one_noisy, atol=1e-10, rtol=1e-8):
                raise ValueError(f"noisy_probabilities['{name}'] does not sum to 1.")
            if not torch.isclose(torch.sum(empirical), one_emp, atol=1e-10, rtol=1e-8):
                raise ValueError(f"empirical_probabilities['{name}'] does not sum to 1.")

    def summary(self) -> Dict[str, object]:
        """
        Return a compact summary dictionary.
        """
        return {
            "experiment_name": self.config.experiment_name,
            "num_regions": self.config.num_regions,
            "region_names": tuple(region.name for region in self.config.regions),
            "has_global_state": self.global_state is not None,
            "seed": self.config.simulation.seed,
            "use_shot_noise": self.config.simulation.use_shot_noise,
            "device": str(self.config.device),
            "real_dtype": str(self.config.torch_real_dtype),
            "complex_dtype": str(self.config.torch_complex_dtype),
        }

    def pretty_print(self) -> None:
        """
        Print a readable summary of the simulation result.
        """
        print("=" * 72)
        print("SimulationResult summary")
        print("-" * 72)
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Has global state: {self.global_state is not None}")
        print(f"Seed: {self.config.simulation.seed}")
        print(f"Use shot noise: {self.config.simulation.use_shot_noise}")
        print(f"Device: {self.config.device}")
        print("-" * 72)
        for region in self.config.regions:
            name = region.name
            m = self.region_povms[name].num_outcomes
            shots = self.region_shots[name]
            print(
                f"[{name}] shots={shots}, dim={self.region_povms[name].dim}, "
                f"outcomes={m}, counts_sum={int(torch.sum(self.counts[name]).item())}"
            )
        print("=" * 72)


# ============================================================
# Sampling helpers
# ============================================================

def sample_counts_from_probabilities(
    probabilities: torch.Tensor,
    shots: int,
    rng: Optional[Union[int, torch.Generator]] = None,
) -> torch.Tensor:
    """
    Sample multinomial counts from a probability vector.
    """
    p = _normalize_probability_vector(probabilities, "probabilities")
    shots = _ensure_positive_int(shots, "shots")
    gen = _coerce_rng(rng, device=p.device)

    sampled = torch.multinomial(
        p,
        num_samples=shots,
        replacement=True,
        generator=gen,
    )
    counts = torch.bincount(sampled, minlength=p.numel()).to(dtype=torch.int64, device=p.device)
    return counts


def counts_to_empirical_probabilities(counts: torch.Tensor) -> torch.Tensor:
    """
    Convert a count vector to empirical frequencies.
    """
    counts = _as_torch_tensor(counts).reshape(-1)
    if counts.numel() == 0:
        raise ValueError("counts must be non-empty.")
    if torch.any(counts < 0):
        raise ValueError("counts must be non-negative.")
    total = int(torch.sum(counts).item())
    if total <= 0:
        raise ValueError("counts must sum to a positive integer.")

    real_dtype = torch.float64
    if counts.device.type == "cuda":
        real_dtype = torch.float64
    return counts.to(dtype=real_dtype) / float(total)


# ============================================================
# Region-level simulation
# ============================================================

def simulate_region_observation(
    rho: torch.Tensor,
    povm: POVM,
    confusion: torch.Tensor,
    shots: int,
    use_shot_noise: bool = True,
    rng: Optional[Union[int, torch.Generator]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Simulate one region's ideal probabilities, noisy probabilities, counts,
    and empirical frequencies.
    """
    shots = _ensure_positive_int(shots, "shots")
    gen = _coerce_rng(rng, device=povm.device)

    rho_t = _as_torch_tensor(rho, dtype=povm.dtype, device=povm.device)
    confusion_t = _as_torch_tensor(
        confusion,
        dtype=_real_dtype_for_complex(povm.dtype),
        device=povm.device,
    )

    ideal = measurement_map(rho_t, povm)
    noisy = apply_confusion_matrix(confusion_t, ideal, prob_floor=0.0)

    if use_shot_noise:
        counts = sample_counts_from_probabilities(noisy, shots=shots, rng=gen)
        empirical = counts_to_empirical_probabilities(counts).to(dtype=noisy.dtype, device=noisy.device)
    else:
        empirical = noisy.clone()
        counts = torch.round(shots * empirical).to(dtype=torch.int64)

        diff = shots - int(torch.sum(counts).item())
        if diff != 0:
            idx = int(torch.argmax(empirical).item())
            counts[idx] += diff

    return {
        "ideal_probabilities": ideal,
        "noisy_probabilities": noisy,
        "counts": counts,
        "empirical_probabilities": empirical,
    }


# ============================================================
# Full experiment simulation
# ============================================================

def simulate_experiment(
    cfg: ExperimentConfig,
    *,
    site_model_override: Optional[str] = None,
    truth_mode: str = "global_consistent",
    independent_rank: Optional[int] = None,
    confusion_dirichlet_concentration: float = 1.0,
) -> SimulationResult:
    """
    Simulate a complete synthetic experiment.

    Parameters
    ----------
    cfg :
        Experiment configuration.
    site_model_override :
        Optional override for the site-level truth model when using
        truth_mode='global_consistent'.
    truth_mode :
        One of:
        - 'global_consistent'
        - 'independent_regions'
    independent_rank :
        Optional rank for independently generated random mixed states.
    confusion_dirichlet_concentration :
        Concentration parameter used when a true confusion model is
        random_column_stochastic.

    Returns
    -------
    SimulationResult
        Complete synthetic experiment bundle.

    Notes
    -----
    The recommended mode for the full project is 'global_consistent', since it
    guarantees overlap-consistent regional truth states.
    """
    truth_mode = str(truth_mode)
    if truth_mode not in {"global_consistent", "independent_regions"}:
        raise ValueError(
            f"truth_mode must be one of {{'global_consistent', 'independent_regions'}}, got '{truth_mode}'."
        )

    cfg.apply_runtime()
    cfg.simulation.seed_all(cfg.runtime)
    rng = cfg.make_torch_generator()

    # --------------------------------------------------------
    # Truth states
    # --------------------------------------------------------
    if truth_mode == "global_consistent":
        site_model = _resolve_global_site_model(cfg, site_model_override)
        global_state, site_states, region_states = generate_consistent_regional_truth_from_global_product(
            cfg=cfg,
            site_model=site_model,
            rng=rng,
            rank=cfg.simulation.state_rank,
        )
        validate_region_state_collection(cfg, region_states, check_overlap_consistency=True)
    else:
        global_state = None
        site_states = None
        region_states = generate_independent_regional_truth(
            cfg=cfg,
            rng=rng,
            rank=independent_rank if independent_rank is not None else cfg.simulation.state_rank,
        )
        validate_region_state_collection(cfg, region_states, check_overlap_consistency=False)

    # --------------------------------------------------------
    # POVMs and true confusions
    # --------------------------------------------------------
    region_povms = build_all_region_povms(cfg, rng=rng)
    region_confusions = build_all_true_confusions(
        cfg,
        rng=rng,
        concentration=confusion_dirichlet_concentration,
    )

    validate_region_povm_collection(cfg, region_povms)
    validate_region_confusion_collection(cfg, region_confusions)

    # --------------------------------------------------------
    # Region-wise observation generation
    # --------------------------------------------------------
    ideal_probabilities: Dict[str, torch.Tensor] = {}
    noisy_probabilities: Dict[str, torch.Tensor] = {}
    counts: Dict[str, torch.Tensor] = {}
    empirical_probabilities: Dict[str, torch.Tensor] = {}
    region_shots: Dict[str, int] = {}

    for region in cfg.regions:
        name = region.name
        shots = int(region.shots)
        region_shots[name] = shots

        record = simulate_region_observation(
            rho=region_states[name],
            povm=region_povms[name],
            confusion=region_confusions[name],
            shots=shots,
            use_shot_noise=cfg.simulation.use_shot_noise,
            rng=rng,
        )

        ideal_probabilities[name] = record["ideal_probabilities"]
        noisy_probabilities[name] = record["noisy_probabilities"]
        counts[name] = record["counts"]
        empirical_probabilities[name] = record["empirical_probabilities"]

    result = SimulationResult(
        config=cfg,
        global_state=global_state,
        site_states=site_states,
        region_states=region_states,
        region_povms=region_povms,
        region_confusions=region_confusions,
        ideal_probabilities=ideal_probabilities,
        noisy_probabilities=noisy_probabilities,
        counts=counts,
        empirical_probabilities=empirical_probabilities,
        region_shots=region_shots,
        metadata={
            "truth_mode": truth_mode,
            "site_model_override": site_model_override,
            "confusion_dirichlet_concentration": float(confusion_dirichlet_concentration),
        },
    )
    result.validate()
    return result


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_count_sampling() -> None:
    p = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    counts = sample_counts_from_probabilities(p, shots=1000, rng=123)
    empirical = counts_to_empirical_probabilities(counts)

    assert tuple(counts.shape) == (3,)
    assert int(torch.sum(counts).item()) == 1000
    assert tuple(empirical.shape) == (3,)
    assert torch.isclose(
        torch.sum(empirical),
        torch.tensor(1.0, dtype=empirical.dtype, device=empirical.device),
        atol=1e-12,
    )


def _self_test_region_simulation() -> None:
    from measurements import make_computational_povm
    from noise import make_noisy_identity_confusion

    rho = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128)
    povm = make_computational_povm(2)
    confusion = make_noisy_identity_confusion(2, strength=0.2)

    record = simulate_region_observation(
        rho=rho,
        povm=povm,
        confusion=confusion,
        shots=500,
        use_shot_noise=True,
        rng=123,
    )

    assert tuple(record["ideal_probabilities"].shape) == (2,)
    assert tuple(record["noisy_probabilities"].shape) == (2,)
    assert tuple(record["counts"].shape) == (2,)
    assert tuple(record["empirical_probabilities"].shape) == (2,)
    assert int(torch.sum(record["counts"]).item()) == 500
    assert torch.isclose(
        torch.sum(record["ideal_probabilities"]),
        torch.tensor(1.0, dtype=record["ideal_probabilities"].dtype, device=record["ideal_probabilities"].device),
        atol=1e-12,
    )
    assert torch.isclose(
        torch.sum(record["noisy_probabilities"]),
        torch.tensor(1.0, dtype=record["noisy_probabilities"].dtype, device=record["noisy_probabilities"].device),
        atol=1e-12,
    )
    assert torch.isclose(
        torch.sum(record["empirical_probabilities"]),
        torch.tensor(1.0, dtype=record["empirical_probabilities"].dtype, device=record["empirical_probabilities"].device),
        atol=1e-12,
    )


def _self_test_full_simulation_global_consistent() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    result = simulate_experiment(cfg, truth_mode="global_consistent")
    result.validate()

    assert result.global_state is not None
    assert result.site_states is not None
    assert set(result.region_states.keys()) == {region.name for region in cfg.regions}
    assert set(result.region_povms.keys()) == {region.name for region in cfg.regions}
    assert set(result.region_confusions.keys()) == {region.name for region in cfg.regions}


def _self_test_full_simulation_independent_regions() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    result = simulate_experiment(cfg, truth_mode="independent_regions")
    result.validate()

    assert result.global_state is None
    assert result.site_states is None
    assert set(result.region_states.keys()) == {region.name for region in cfg.regions}


def _self_test_identity_noiseless_fixed_point() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    for region in cfg.regions:
        region.true_confusion_model = "identity"

    result = simulate_experiment(cfg, truth_mode="global_consistent")
    result.validate()

    for region in cfg.regions:
        name = region.name
        assert torch.allclose(
            result.empirical_probabilities[name],
            result.noisy_probabilities[name],
            atol=1e-12,
        )


def _self_test_gpu_smoke() -> None:
    if not torch.cuda.is_available():
        return

    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    cfg.runtime.device = "cuda"
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = True

    result = simulate_experiment(cfg, truth_mode="global_consistent")
    result.validate()

    assert result.region_states[cfg.regions[0].name].device.type == "cuda"
    assert result.region_confusions[cfg.regions[0].name].device.type == "cuda"
    assert result.ideal_probabilities[cfg.regions[0].name].device.type == "cuda"
    assert result.noisy_probabilities[cfg.regions[0].name].device.type == "cuda"
    assert result.counts[cfg.regions[0].name].device.type == "cuda"


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the simulator module.
    """
    tests = [
        ("count sampling", _self_test_count_sampling),
        ("region simulation", _self_test_region_simulation),
        ("full simulation (global consistent)", _self_test_full_simulation_global_consistent),
        ("full simulation (independent regions)", _self_test_full_simulation_independent_regions),
        ("identity noiseless fixed point", _self_test_identity_noiseless_fixed_point),
        ("gpu smoke", _self_test_gpu_smoke),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All simulator self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
