from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ============================================================
# Supported option sets
# ============================================================

SUPPORTED_LOSSES = {"l2", "nll"}
SUPPORTED_POVM_TYPES = {"random_ic", "computational", "pauli6_single_qubit"}
SUPPORTED_STATE_INIT_METHODS = {"maximally_mixed", "random_mixed", "random_pure"}
SUPPORTED_CONFUSION_INIT_METHODS = {"identity", "uniform", "noisy_identity"}
SUPPORTED_TRUE_STATE_MODELS = {"random_mixed", "random_pure", "maximally_mixed"}
SUPPORTED_TRUE_CONFUSION_MODELS = {"identity", "noisy_identity", "random_column_stochastic"}
SUPPORTED_REAL_DTYPES = {"float32", "float64"}
SUPPORTED_COMPLEX_DTYPES = {"complex64", "complex128"}
SUPPORTED_FLOAT32_MATMUL_PRECISION = {"highest", "high", "medium"}

_REAL_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}
_COMPLEX_DTYPE_MAP: Dict[str, torch.dtype] = {
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
_COMPLEX_FOR_REAL: Dict[str, str] = {
    "float32": "complex64",
    "float64": "complex128",
}


# ============================================================
# Small validation helpers
# ============================================================

def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _ensure_nonnegative_int(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
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


def _ensure_string_choice(value: str, allowed: set[str], name: str) -> str:
    value = str(value)
    if value not in allowed:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of {{{allowed_str}}}, got '{value}'.")
    return value


def _normalize_sites(sites: Sequence[int], name: str = "sites") -> Tuple[int, ...]:
    if len(sites) == 0:
        raise ValueError(f"{name} must be non-empty.")
    normalized = tuple(int(s) for s in sites)
    if any(s < 0 for s in normalized):
        raise ValueError(f"All entries in {name} must be non-negative. Got {normalized}.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate indices. Got {normalized}.")
    return normalized


def _validate_torch_device_string(device: str, fallback_to_cpu: bool) -> str:
    device = str(device).strip().lower()
    if len(device) == 0:
        raise ValueError("device must be a non-empty string.")

    if device == "auto":
        return device

    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid torch device specification '{device}'.") from exc

    if parsed.type == "cuda" and not torch.cuda.is_available() and not fallback_to_cpu:
        raise ValueError(
            f"Requested CUDA device '{device}', but CUDA is not available and fallback_to_cpu=False."
        )

    return device


# ============================================================
# Runtime / backend configuration
# ============================================================

@dataclass
class RuntimeConfig:
    """
    Runtime configuration for CPU / GPU execution.

    Notes
    -----
    - `device='auto'` chooses CUDA when available, otherwise CPU.
    - Defaults keep double precision for numerical stability and fidelity with
      the original NumPy implementation.
    - If you want maximum throughput later, you can switch to
      float32 / complex64 once the full pipeline is validated.
    """

    device: str = "auto"
    real_dtype: str = "float64"
    complex_dtype: str = "complex128"
    fallback_to_cpu: bool = True
    deterministic: bool = False
    float32_matmul_precision: str = "high"
    num_threads: Optional[int] = None

    def __post_init__(self) -> None:
        self.device = _validate_torch_device_string(self.device, bool(self.fallback_to_cpu))
        self.real_dtype = _ensure_string_choice(self.real_dtype, SUPPORTED_REAL_DTYPES, "real_dtype")
        self.complex_dtype = _ensure_string_choice(
            self.complex_dtype,
            SUPPORTED_COMPLEX_DTYPES,
            "complex_dtype",
        )
        self.fallback_to_cpu = bool(self.fallback_to_cpu)
        self.deterministic = bool(self.deterministic)
        self.float32_matmul_precision = _ensure_string_choice(
            self.float32_matmul_precision,
            SUPPORTED_FLOAT32_MATMUL_PRECISION,
            "float32_matmul_precision",
        )
        if self.num_threads is not None:
            self.num_threads = _ensure_positive_int(self.num_threads, "num_threads")

        expected_complex = _COMPLEX_FOR_REAL[self.real_dtype]
        if self.complex_dtype != expected_complex:
            raise ValueError(
                f"complex_dtype must match real_dtype. For real_dtype='{self.real_dtype}', "
                f"use complex_dtype='{expected_complex}', got '{self.complex_dtype}'."
            )

    def resolve_device(self) -> torch.device:
        """Resolve the effective torch device, honoring auto-selection and fallback."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        parsed = torch.device(self.device)
        if parsed.type == "cuda" and not torch.cuda.is_available():
            if self.fallback_to_cpu:
                return torch.device("cpu")
            raise RuntimeError(f"Requested CUDA device '{self.device}', but CUDA is unavailable.")
        return parsed

    @property
    def torch_real_dtype(self) -> torch.dtype:
        """Torch real dtype corresponding to `real_dtype`."""
        return _REAL_DTYPE_MAP[self.real_dtype]

    @property
    def torch_complex_dtype(self) -> torch.dtype:
        """Torch complex dtype corresponding to `complex_dtype`."""
        return _COMPLEX_DTYPE_MAP[self.complex_dtype]

    def apply(self) -> None:
        """
        Apply global torch runtime knobs.

        This is intentionally separate from construction so config creation does
        not silently mutate global process state.
        """
        if self.num_threads is not None:
            torch.set_num_threads(self.num_threads)

        torch.use_deterministic_algorithms(self.deterministic)

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(self.float32_matmul_precision)

    def summary_dict(self) -> Dict[str, Any]:
        effective = self.resolve_device()
        return {
            "requested_device": self.device,
            "effective_device": str(effective),
            "real_dtype": self.real_dtype,
            "complex_dtype": self.complex_dtype,
            "fallback_to_cpu": self.fallback_to_cpu,
            "deterministic": self.deterministic,
            "float32_matmul_precision": self.float32_matmul_precision,
            "num_threads": self.num_threads,
        }


# ============================================================
# Region configuration
# ============================================================

@dataclass
class RegionConfig:
    """
    Configuration for one region in the multi-region formulation.

    Parameters
    ----------
    name :
        Region identifier.
    sites :
        Site indices included in the region.
    shots :
        Number of measurement shots used for this region.
    povm_type :
        Type of POVM to construct in the measurement module.
    povm_num_outcomes :
        Number of POVM outcomes. If None, the measurement module may choose
        a default based on the region dimension and POVM type.
    true_state_model :
        Synthetic ground-truth regional state model used by the simulator.
    init_state_method :
        Initialization method for the regional state estimate.
    true_confusion_model :
        Synthetic ground-truth confusion model used by the simulator.
    init_confusion_method :
        Initialization method for the regional confusion estimate.
    confusion_strength :
        Noise strength parameter for synthetic confusion-matrix generation.
    reference_confusion_type :
        Reference confusion model used inside regularization.
        Usually 'identity'.
    weight :
        Optional region weight for future extensions.
    """

    name: str
    sites: Tuple[int, ...]
    shots: int = 2000

    povm_type: str = "random_ic"
    povm_num_outcomes: Optional[int] = None

    true_state_model: str = "random_mixed"
    init_state_method: str = "maximally_mixed"

    true_confusion_model: str = "identity"
    init_confusion_method: str = "identity"
    confusion_strength: float = 0.05

    reference_confusion_type: str = "identity"
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if len(self.name) == 0:
            raise ValueError("Region name must be a non-empty string.")

        self.sites = _normalize_sites(self.sites, name=f"sites for region '{self.name}'")
        self.shots = _ensure_positive_int(self.shots, f"shots for region '{self.name}'")

        self.povm_type = _ensure_string_choice(
            self.povm_type,
            SUPPORTED_POVM_TYPES,
            f"povm_type for region '{self.name}'",
        )

        if self.povm_num_outcomes is not None:
            self.povm_num_outcomes = _ensure_positive_int(
                self.povm_num_outcomes,
                f"povm_num_outcomes for region '{self.name}'",
            )

        self.true_state_model = _ensure_string_choice(
            self.true_state_model,
            SUPPORTED_TRUE_STATE_MODELS,
            f"true_state_model for region '{self.name}'",
        )
        self.init_state_method = _ensure_string_choice(
            self.init_state_method,
            SUPPORTED_STATE_INIT_METHODS,
            f"init_state_method for region '{self.name}'",
        )
        self.true_confusion_model = _ensure_string_choice(
            self.true_confusion_model,
            SUPPORTED_TRUE_CONFUSION_MODELS,
            f"true_confusion_model for region '{self.name}'",
        )
        self.init_confusion_method = _ensure_string_choice(
            self.init_confusion_method,
            SUPPORTED_CONFUSION_INIT_METHODS,
            f"init_confusion_method for region '{self.name}'",
        )

        self.confusion_strength = _ensure_nonnegative_float(
            self.confusion_strength,
            f"confusion_strength for region '{self.name}'",
        )
        self.weight = _ensure_positive_float(self.weight, f"weight for region '{self.name}'")

        if self.reference_confusion_type not in {"identity"}:
            raise ValueError(
                f"reference_confusion_type for region '{self.name}' must currently be 'identity', "
                f"got '{self.reference_confusion_type}'."
            )

    @property
    def num_sites(self) -> int:
        """Number of sites included in the region."""
        return len(self.sites)


# ============================================================
# Loss / discrepancy configuration
# ============================================================

@dataclass
class LossConfig:
    """
    Configuration for the discrepancy function used in fitting.

    Supported losses
    ----------------
    l2 :
        Squared Euclidean discrepancy.
    nll :
        Multinomial negative log-likelihood using observed counts.

    Notes
    -----
    `prob_floor` is used to avoid log(0) and similar numerical issues.
    """

    name: str = "nll"
    prob_floor: float = 1e-12

    def __post_init__(self) -> None:
        self.name = _ensure_string_choice(self.name, SUPPORTED_LOSSES, "loss name")
        self.prob_floor = _ensure_positive_float(self.prob_floor, "prob_floor")


# ============================================================
# Solver configuration
# ============================================================

@dataclass
class ADMMConfig:
    """
    Configuration for the outer alternating loop and inner ADMM state solver.

    Main parameters
    ---------------
    beta :
        Augmented-Lagrangian penalty for overlap consensus in the inner ADMM loop.
    gamma_rho :
        Proximal weight for the regional state update.
    gamma_c :
        Proximal weight for the confusion-matrix update.
    lambda_confusion :
        Regularization weight on confusion matrices.
    outer_max_iters :
        Maximum number of outer alternating iterations.
    inner_max_iters :
        Maximum number of inner ADMM iterations per outer step.
    """

    beta: float = 1.0
    gamma_rho: float = 1.0
    gamma_c: float = 1.0
    lambda_confusion: float = 1e-2

    outer_max_iters: int = 50
    inner_max_iters: int = 100

    outer_tol: float = 1e-6
    inner_primal_tol: float = 1e-6
    inner_dual_tol: float = 1e-6

    state_step_size: float = 0.1
    confusion_step_size: float = 0.1

    state_gd_max_iters: int = 200
    confusion_gd_max_iters: int = 200

    state_gd_tol: float = 1e-8
    confusion_gd_tol: float = 1e-8

    verbose: bool = True
    store_history: bool = True
    print_every: int = 1

    def __post_init__(self) -> None:
        self.beta = _ensure_positive_float(self.beta, "beta")
        self.gamma_rho = _ensure_positive_float(self.gamma_rho, "gamma_rho")
        self.gamma_c = _ensure_positive_float(self.gamma_c, "gamma_c")
        self.lambda_confusion = _ensure_nonnegative_float(
            self.lambda_confusion,
            "lambda_confusion",
        )

        self.outer_max_iters = _ensure_positive_int(self.outer_max_iters, "outer_max_iters")
        self.inner_max_iters = _ensure_positive_int(self.inner_max_iters, "inner_max_iters")

        self.outer_tol = _ensure_positive_float(self.outer_tol, "outer_tol")
        self.inner_primal_tol = _ensure_positive_float(self.inner_primal_tol, "inner_primal_tol")
        self.inner_dual_tol = _ensure_positive_float(self.inner_dual_tol, "inner_dual_tol")

        self.state_step_size = _ensure_positive_float(self.state_step_size, "state_step_size")
        self.confusion_step_size = _ensure_positive_float(
            self.confusion_step_size,
            "confusion_step_size",
        )

        self.state_gd_max_iters = _ensure_positive_int(
            self.state_gd_max_iters,
            "state_gd_max_iters",
        )
        self.confusion_gd_max_iters = _ensure_positive_int(
            self.confusion_gd_max_iters,
            "confusion_gd_max_iters",
        )

        self.state_gd_tol = _ensure_positive_float(self.state_gd_tol, "state_gd_tol")
        self.confusion_gd_tol = _ensure_positive_float(
            self.confusion_gd_tol,
            "confusion_gd_tol",
        )

        self.verbose = bool(self.verbose)
        self.store_history = bool(self.store_history)
        self.print_every = _ensure_positive_int(self.print_every, "print_every")


# ============================================================
# Simulation configuration
# ============================================================

@dataclass
class SimulationConfig:
    """
    Configuration for synthetic-data generation.

    Parameters
    ----------
    seed :
        Random seed for reproducibility.
    use_shot_noise :
        If True, observed counts are sampled from a multinomial model.
        If False, empirical frequencies are set equal to the noisy probabilities.
    state_rank :
        Optional target rank used by later random-state constructors.
        If None, the simulator chooses a default.
    enforce_physical_truth :
        Whether generated truth objects must be projected / validated.
    """

    seed: int = 12345
    use_shot_noise: bool = True
    state_rank: Optional[int] = None
    enforce_physical_truth: bool = True

    def __post_init__(self) -> None:
        self.seed = int(self.seed)
        self.use_shot_noise = bool(self.use_shot_noise)
        self.enforce_physical_truth = bool(self.enforce_physical_truth)
        if self.state_rank is not None:
            self.state_rank = _ensure_positive_int(self.state_rank, "state_rank")

    def make_rng(self) -> np.random.Generator:
        """
        Create a NumPy random generator from the stored seed.

        Kept for compatibility while older NumPy-oriented modules are still
        being replaced. GPU-adapted modules should use `make_torch_generator`.
        """
        return np.random.default_rng(self.seed)

    def make_numpy_rng(self) -> np.random.Generator:
        """Explicit NumPy RNG constructor."""
        return np.random.default_rng(self.seed)

    def make_torch_generator(self, runtime: Optional[RuntimeConfig] = None) -> torch.Generator:
        """Create a torch.Generator on the effective runtime device."""
        target = torch.device("cpu") if runtime is None else runtime.resolve_device()
        generator = torch.Generator(device=target.type)
        generator.manual_seed(self.seed)
        return generator

    def seed_all(self, runtime: Optional[RuntimeConfig] = None) -> None:
        """Seed NumPy and torch global RNG streams for reproducibility."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        target = torch.device("cpu") if runtime is None else runtime.resolve_device()
        if target.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


# ============================================================
# Top-level experiment configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    This object is the main configuration entry point used by later modules.

    Parameters
    ----------
    qubits_per_site :
        Number of qubits stored at each site.
    regions :
        Regional decomposition of the system.
    loss :
        Discrepancy configuration.
    admm :
        Outer alternating / inner ADMM solver configuration.
    simulation :
        Synthetic-data generation configuration.
    runtime :
        CPU / GPU backend configuration shared across all modules.
    experiment_name :
        Human-readable label.
    """

    qubits_per_site: Tuple[int, ...]
    regions: Tuple[RegionConfig, ...]
    loss: LossConfig = field(default_factory=LossConfig)
    admm: ADMMConfig = field(default_factory=ADMMConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment_name: str = "default_experiment"

    def __post_init__(self) -> None:
        self.experiment_name = str(self.experiment_name).strip()
        if len(self.experiment_name) == 0:
            raise ValueError("experiment_name must be a non-empty string.")

        self.qubits_per_site = tuple(
            _ensure_positive_int(q, "each entry of qubits_per_site")
            for q in self.qubits_per_site
        )
        if len(self.qubits_per_site) == 0:
            raise ValueError("qubits_per_site must be non-empty.")

        self.regions = tuple(self.regions)
        if len(self.regions) == 0:
            raise ValueError("At least one region must be provided.")

        if len({region.name for region in self.regions}) != len(self.regions):
            names = [region.name for region in self.regions]
            raise ValueError(f"Region names must be unique. Got {names}.")

        self._validate_region_sites()
        self._validate_region_dimensions()
        self._validate_povm_outcome_counts()

    # --------------------------------------------------------
    # Basic structural properties
    # --------------------------------------------------------

    @property
    def num_sites(self) -> int:
        """Total number of sites."""
        return len(self.qubits_per_site)

    @property
    def total_qubits(self) -> int:
        """Total number of qubits across all sites."""
        return int(sum(self.qubits_per_site))

    @property
    def site_dimensions(self) -> Tuple[int, ...]:
        """Hilbert-space dimensions of the sites."""
        return tuple(2 ** q for q in self.qubits_per_site)

    @property
    def num_regions(self) -> int:
        """Number of regions."""
        return len(self.regions)

    @property
    def device(self) -> torch.device:
        """Effective torch device for this experiment."""
        return self.runtime.resolve_device()

    @property
    def torch_real_dtype(self) -> torch.dtype:
        """Effective torch real dtype for this experiment."""
        return self.runtime.torch_real_dtype

    @property
    def torch_complex_dtype(self) -> torch.dtype:
        """Effective torch complex dtype for this experiment."""
        return self.runtime.torch_complex_dtype

    def apply_runtime(self) -> None:
        """Apply the runtime's global torch settings."""
        self.runtime.apply()

    def make_torch_generator(self) -> torch.Generator:
        """Convenience wrapper for a seeded torch.Generator."""
        return self.simulation.make_torch_generator(self.runtime)

    # --------------------------------------------------------
    # Region-level derived quantities
    # --------------------------------------------------------

    def region_by_name(self, name: str) -> RegionConfig:
        """Return the region configuration with the given name."""
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"No region with name '{name}' was found.")

    def region_index(self, name: str) -> int:
        """Return the index of the named region."""
        for idx, region in enumerate(self.regions):
            if region.name == name:
                return idx
        raise KeyError(f"No region with name '{name}' was found.")

    def region_qubits(self, region: RegionConfig | str) -> int:
        """Total number of qubits in a region."""
        region_obj = self.region_by_name(region) if isinstance(region, str) else region
        return int(sum(self.qubits_per_site[s] for s in region_obj.sites))

    def region_dimension(self, region: RegionConfig | str) -> int:
        """Hilbert-space dimension of a region."""
        return 2 ** self.region_qubits(region)

    def region_site_dimensions(self, region: RegionConfig | str) -> Tuple[int, ...]:
        """Site dimensions within a region."""
        region_obj = self.region_by_name(region) if isinstance(region, str) else region
        return tuple(2 ** self.qubits_per_site[s] for s in region_obj.sites)

    def region_overlap_sites(
        self,
        region_a: RegionConfig | str,
        region_b: RegionConfig | str,
    ) -> Tuple[int, ...]:
        """Sorted tuple of shared site indices between two regions."""
        a = self.region_by_name(region_a) if isinstance(region_a, str) else region_a
        b = self.region_by_name(region_b) if isinstance(region_b, str) else region_b
        return tuple(sorted(set(a.sites).intersection(b.sites)))

    def region_overlap_qubits(
        self,
        region_a: RegionConfig | str,
        region_b: RegionConfig | str,
    ) -> int:
        """Number of qubits in the overlap between two regions."""
        overlap = self.region_overlap_sites(region_a, region_b)
        return int(sum(self.qubits_per_site[s] for s in overlap))

    def overlap_pairs(self) -> Tuple[Tuple[int, int], ...]:
        """
        Return all overlapping region-index pairs (r, r') with r < r'.
        """
        pairs: List[Tuple[int, int]] = []
        for i in range(len(self.regions)):
            for j in range(i + 1, len(self.regions)):
                if len(set(self.regions[i].sites).intersection(self.regions[j].sites)) > 0:
                    pairs.append((i, j))
        return tuple(pairs)

    def neighbors(self, region_index: int) -> Tuple[int, ...]:
        """Indices of regions that overlap with the given region."""
        region_index = _ensure_nonnegative_int(region_index, "region_index")
        if region_index >= self.num_regions:
            raise ValueError(
                f"region_index must be in [0, {self.num_regions - 1}], got {region_index}."
            )
        nbrs = []
        target_sites = set(self.regions[region_index].sites)
        for j, region in enumerate(self.regions):
            if j == region_index:
                continue
            if len(target_sites.intersection(region.sites)) > 0:
                nbrs.append(j)
        return tuple(nbrs)

    # --------------------------------------------------------
    # Validation logic
    # --------------------------------------------------------

    def _validate_region_sites(self) -> None:
        for region in self.regions:
            for s in region.sites:
                if s >= self.num_sites:
                    raise ValueError(
                        f"Region '{region.name}' contains site index {s}, but only "
                        f"{self.num_sites} sites are available."
                    )

    def _validate_region_dimensions(self) -> None:
        for region in self.regions:
            region_dim = self.region_dimension(region)
            if region_dim <= 0:
                raise ValueError(f"Region '{region.name}' has invalid dimension {region_dim}.")

    def _validate_povm_outcome_counts(self) -> None:
        for region in self.regions:
            if region.povm_num_outcomes is None:
                continue

            region_dim = self.region_dimension(region)
            min_informationally_complete = region_dim ** 2

            if region.povm_num_outcomes < 2:
                raise ValueError(
                    f"Region '{region.name}' has povm_num_outcomes={region.povm_num_outcomes}, "
                    f"but at least 2 outcomes are required."
                )

            # We do not force informational completeness for every user experiment,
            # but we do warn via an exception only if the user explicitly picked a
            # value that is impossible to interpret later as IC when they use
            # a nominally informationally complete POVM family.
            if region.povm_type == "random_ic" and region.povm_num_outcomes < min_informationally_complete:
                raise ValueError(
                    f"Region '{region.name}' uses povm_type='random_ic' but "
                    f"povm_num_outcomes={region.povm_num_outcomes} is smaller than "
                    f"dimension^2={min_informationally_complete}. For an IC random POVM, "
                    f"use at least {min_informationally_complete} outcomes."
                )

            if region.povm_type == "computational" and region.povm_num_outcomes != region_dim:
                raise ValueError(
                    f"Region '{region.name}' uses povm_type='computational', so "
                    f"povm_num_outcomes must equal region dimension {region_dim}, got "
                    f"{region.povm_num_outcomes}."
                )

            if region.povm_type == "pauli6_single_qubit":
                if self.region_qubits(region) != 1:
                    raise ValueError(
                        f"Region '{region.name}' uses povm_type='pauli6_single_qubit' but "
                        f"contains {self.region_qubits(region)} qubits. This POVM type is "
                        f"currently restricted to 1-qubit regions."
                    )
                if region.povm_num_outcomes is not None and region.povm_num_outcomes != 6:
                    raise ValueError(
                        f"Region '{region.name}' uses povm_type='pauli6_single_qubit', so "
                        f"povm_num_outcomes must be 6."
                    )

    # --------------------------------------------------------
    # Serialization helpers
    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a fully serializable dictionary representation."""
        return asdict(self)

    def summary_dict(self) -> Dict[str, Any]:
        """Return a compact summary dictionary useful for logging."""
        return {
            "experiment_name": self.experiment_name,
            "num_sites": self.num_sites,
            "total_qubits": self.total_qubits,
            "num_regions": self.num_regions,
            "qubits_per_site": self.qubits_per_site,
            "overlap_pairs": self.overlap_pairs(),
            "loss": self.loss.name,
            "seed": self.simulation.seed,
            "runtime": self.runtime.summary_dict(),
        }

    def pretty_print(self) -> None:
        """Print a readable summary of the experiment configuration."""
        runtime_summary = self.runtime.summary_dict()

        print("=" * 72)
        print(f"Experiment: {self.experiment_name}")
        print("-" * 72)
        print(f"Sites: {self.num_sites}")
        print(f"Qubits per site: {self.qubits_per_site}")
        print(f"Total qubits: {self.total_qubits}")
        print(f"Regions: {self.num_regions}")
        print(f"Loss: {self.loss.name}")
        print(f"Seed: {self.simulation.seed}")
        print(f"Requested device: {runtime_summary['requested_device']}")
        print(f"Effective device: {runtime_summary['effective_device']}")
        print(f"Real dtype: {runtime_summary['real_dtype']}")
        print(f"Complex dtype: {runtime_summary['complex_dtype']}")
        print(f"Overlap pairs: {self.overlap_pairs()}")
        print("-" * 72)
        for idx, region in enumerate(self.regions):
            print(
                f"[Region {idx}] name={region.name}, sites={region.sites}, "
                f"q={self.region_qubits(region)}, dim={self.region_dimension(region)}, "
                f"shots={region.shots}, povm_type={region.povm_type}, "
                f"povm_num_outcomes={region.povm_num_outcomes}"
            )
        print("=" * 72)


# ============================================================
# Convenience builders for common layouts
# ============================================================

def build_sliding_window_regions(
    num_sites: int,
    window_size: int,
    *,
    shots: int = 2000,
    povm_type: str = "random_ic",
    povm_num_outcomes: Optional[int] = None,
    true_state_model: str = "random_mixed",
    init_state_method: str = "maximally_mixed",
    true_confusion_model: str = "identity",
    init_confusion_method: str = "identity",
    confusion_strength: float = 0.05,
    name_prefix: str = "R",
) -> Tuple[RegionConfig, ...]:
    """
    Build regions of the form
        (0, 1, ..., w-1), (1, 2, ..., w), ..., (n-w, ..., n-1)

    Parameters
    ----------
    num_sites :
        Total number of sites.
    window_size :
        Number of consecutive sites per region.

    Returns
    -------
    tuple[RegionConfig, ...]
    """
    num_sites = _ensure_positive_int(num_sites, "num_sites")
    window_size = _ensure_positive_int(window_size, "window_size")
    if window_size > num_sites:
        raise ValueError(
            f"window_size must not exceed num_sites, got window_size={window_size}, "
            f"num_sites={num_sites}."
        )

    regions: List[RegionConfig] = []
    for start in range(num_sites - window_size + 1):
        sites = tuple(range(start, start + window_size))
        region = RegionConfig(
            name=f"{name_prefix}{start}",
            sites=sites,
            shots=shots,
            povm_type=povm_type,
            povm_num_outcomes=povm_num_outcomes,
            true_state_model=true_state_model,
            init_state_method=init_state_method,
            true_confusion_model=true_confusion_model,
            init_confusion_method=init_confusion_method,
            confusion_strength=confusion_strength,
        )
        regions.append(region)
    return tuple(regions)


def build_pairwise_chain_regions(
    num_sites: int,
    *,
    shots: int = 2000,
    povm_type: str = "random_ic",
    povm_num_outcomes: Optional[int] = None,
    true_state_model: str = "random_mixed",
    init_state_method: str = "maximally_mixed",
    true_confusion_model: str = "identity",
    init_confusion_method: str = "identity",
    confusion_strength: float = 0.05,
    name_prefix: str = "R",
) -> Tuple[RegionConfig, ...]:
    """
    Build pairwise chain regions:
        (0,1), (1,2), ..., (n-2,n-1)
    """
    return build_sliding_window_regions(
        num_sites=num_sites,
        window_size=2,
        shots=shots,
        povm_type=povm_type,
        povm_num_outcomes=povm_num_outcomes,
        true_state_model=true_state_model,
        init_state_method=init_state_method,
        true_confusion_model=true_confusion_model,
        init_confusion_method=init_confusion_method,
        confusion_strength=confusion_strength,
        name_prefix=name_prefix,
    )


def make_default_experiment_config() -> ExperimentConfig:
    """
    Create a small default experiment configuration that is safe for debugging.

    Default setup
    -------------
    - 3 sites
    - 1 qubit per site
    - pairwise overlapping regions: (0,1), (1,2)
    - random informationally complete POVMs
    - multinomial negative log-likelihood loss
    - GPU-ready runtime config with automatic device selection
    """
    qubits_per_site = (1, 1, 1)
    regions = build_pairwise_chain_regions(
        num_sites=3,
        shots=1000,
        povm_type="random_ic",
        povm_num_outcomes=16,  # 2 qubits -> dim=4 -> dim^2=16
        true_state_model="random_mixed",
        init_state_method="maximally_mixed",
        true_confusion_model="noisy_identity",
        init_confusion_method="identity",
        confusion_strength=0.03,
        name_prefix="R",
    )

    return ExperimentConfig(
        qubits_per_site=qubits_per_site,
        regions=regions,
        loss=LossConfig(name="nll", prob_floor=1e-12),
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
            seed=12345,
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
        experiment_name="default_pairwise_chain",
    )


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_default_config() -> None:
    cfg = make_default_experiment_config()
    assert cfg.num_sites == 3
    assert cfg.total_qubits == 3
    assert cfg.num_regions == 2
    assert cfg.overlap_pairs() == ((0, 1),)
    assert cfg.region_overlap_sites("R0", "R1") == (1,)
    assert cfg.region_overlap_qubits("R0", "R1") == 1
    assert cfg.region_dimension("R0") == 4
    assert cfg.region_dimension("R1") == 4
    assert cfg.torch_real_dtype == torch.float64
    assert cfg.torch_complex_dtype == torch.complex128
    assert isinstance(cfg.device, torch.device)


def _self_test_sliding_window_builder() -> None:
    regions = build_sliding_window_regions(
        num_sites=4,
        window_size=3,
        shots=500,
        povm_type="computational",
        povm_num_outcomes=8,
    )
    assert len(regions) == 2
    assert regions[0].sites == (0, 1, 2)
    assert regions[1].sites == (1, 2, 3)


def _self_test_runtime_config_validation() -> None:
    runtime = RuntimeConfig(device="auto", real_dtype="float32", complex_dtype="complex64")
    runtime.apply()
    assert runtime.torch_real_dtype == torch.float32
    assert runtime.torch_complex_dtype == torch.complex64


def run_self_tests(verbose: bool = True) -> None:
    tests = [
        ("default config", _self_test_default_config),
        ("sliding-window builder", _self_test_sliding_window_builder),
        ("runtime config", _self_test_runtime_config_validation),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All config self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
    cfg = make_default_experiment_config()
    cfg.pretty_print()
