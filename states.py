from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from config import ExperimentConfig, RegionConfig
from core_ops import (
    frobenius_norm,
    is_density_matrix,
    kron_all,
    maximally_mixed,
    partial_trace,
    project_to_density_matrix,
    subsystem_dimensions_from_qubits,
)


# ============================================================
# Type aliases
# ============================================================

RNGInput = Optional[Union[int, np.random.Generator, torch.Generator]]


# ============================================================
# Internal helpers
# ============================================================

def _coerce_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _coerce_complex_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        return torch.complex128
    if dtype not in {torch.complex64, torch.complex128}:
        raise ValueError(
            f"dtype must be torch.complex64 or torch.complex128, got {dtype}."
        )
    return dtype


def _real_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise ValueError(f"Unsupported complex dtype: {dtype}.")


def _as_tensor(
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


def _ensure_valid_rank(rank: Optional[int], dim: int) -> Optional[int]:
    if rank is None:
        return None
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"rank must be positive when provided, got {rank}.")
    if rank > dim:
        raise ValueError(f"rank={rank} cannot exceed dimension={dim}.")
    return rank


def _region_obj(region: Union[RegionConfig, str], cfg: ExperimentConfig) -> RegionConfig:
    if isinstance(region, RegionConfig):
        return region
    return cfg.region_by_name(region)


def _default_rank_from_cfg(cfg: Optional[ExperimentConfig], rank: Optional[int]) -> Optional[int]:
    if rank is not None:
        return rank
    if cfg is None:
        return None
    return cfg.simulation.state_rank


def _randn_real(
    shape: Sequence[int] | torch.Size,
    *,
    rng: RNGInput = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    device = _coerce_device(device)

    if isinstance(rng, np.random.Generator):
        arr = rng.normal(size=tuple(shape))
        return torch.as_tensor(arr, dtype=dtype, device=device)

    if isinstance(rng, int):
        gen = torch.Generator(device=device.type)
        gen.manual_seed(int(rng))
        return torch.randn(*tuple(shape), dtype=dtype, device=device, generator=gen)

    if isinstance(rng, torch.Generator):
        return torch.randn(*tuple(shape), dtype=dtype, device=device, generator=rng)

    return torch.randn(*tuple(shape), dtype=dtype, device=device)


def _randn_complex(
    shape: Sequence[int] | torch.Size,
    *,
    rng: RNGInput = None,
    dtype: torch.dtype = torch.complex128,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    dtype = _coerce_complex_dtype(dtype)
    real_dtype = _real_dtype_for_complex(dtype)
    real = _randn_real(shape, rng=rng, dtype=real_dtype, device=device)
    imag = _randn_real(shape, rng=rng, dtype=real_dtype, device=device)
    return real + 1j * imag


# ============================================================
# State-vector and density-matrix primitives
# ============================================================

def normalize_state_vector(psi: torch.Tensor, atol: float = 1e-14) -> torch.Tensor:
    """
    Normalize a complex state vector to unit Euclidean norm.
    """
    psi = _as_tensor(psi, dtype=torch.complex128 if not isinstance(psi, torch.Tensor) else None).reshape(-1)
    if psi.numel() == 0:
        raise ValueError("psi must be non-empty.")
    norm = torch.linalg.vector_norm(psi)
    if float(norm.item()) <= atol:
        raise ValueError("Cannot normalize a numerically zero vector.")
    return psi / norm


def ket_to_density(psi: torch.Tensor) -> torch.Tensor:
    """
    Convert a state vector into a rank-1 density matrix.
    """
    psi = normalize_state_vector(psi)
    return torch.outer(psi, psi.conj())


def computational_basis_ket(
    index: int,
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Return the computational-basis vector e_index in C^dim.
    """
    index = int(index)
    dim = _ensure_positive_int(dim, "dim")
    dtype = _coerce_complex_dtype(dtype)
    device = _coerce_device(device)

    if index < 0 or index >= dim:
        raise ValueError(f"index must lie in [0, {dim - 1}], got {index}.")

    psi = torch.zeros(dim, dtype=dtype, device=device)
    psi[index] = 1.0
    return psi


def random_complex_vector(
    dim: int,
    rng: RNGInput = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Sample a complex Gaussian vector in C^dim.
    """
    dim = _ensure_positive_int(dim, "dim")
    dtype = _coerce_complex_dtype(dtype)
    device = _coerce_device(device)
    return _randn_complex((dim,), rng=rng, dtype=dtype, device=device)


def random_pure_state_ket(
    dim: int,
    rng: RNGInput = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Sample a random pure-state ket in C^dim using normalized complex Gaussian entries.
    """
    dim = _ensure_positive_int(dim, "dim")
    psi = random_complex_vector(dim, rng=rng, dtype=dtype, device=device)
    return normalize_state_vector(psi)


def random_pure_density_matrix(
    dim: int,
    rng: RNGInput = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Sample a random pure-state density matrix in dimension `dim`.
    """
    psi = random_pure_state_ket(dim, rng=rng, dtype=dtype, device=device)
    return ket_to_density(psi)


def random_mixed_density_matrix(
    dim: int,
    rng: RNGInput = None,
    rank: Optional[int] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Sample a random mixed density matrix in dimension `dim`.

    Construction
    ------------
    Let G be a complex Gaussian matrix of shape (dim, rank). Then
        rho = G G^dagger / Tr(G G^dagger).
    """
    dim = _ensure_positive_int(dim, "dim")
    rank = _ensure_valid_rank(rank, dim)
    if rank is None:
        rank = dim

    dtype = _coerce_complex_dtype(dtype)
    device = _coerce_device(device)

    g = _randn_complex((dim, rank), rng=rng, dtype=dtype, device=device)
    rho = g @ g.conj().transpose(-2, -1)
    tr = torch.trace(rho)
    if abs(tr.item()) <= 1e-14:
        raise ValueError("Random mixed-state construction produced a numerically zero trace.")
    rho = rho / tr
    rho = project_to_density_matrix(rho)
    return rho


def sample_density_matrix(
    dim: int,
    model: str = "random_mixed",
    rng: RNGInput = None,
    rank: Optional[int] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Sample a density matrix according to the requested model.

    Supported models
    ----------------
    - "random_mixed"
    - "random_pure"
    - "maximally_mixed"
    """
    dim = _ensure_positive_int(dim, "dim")
    model = str(model)
    dtype = _coerce_complex_dtype(dtype)
    device = _coerce_device(device)

    if model == "random_mixed":
        return random_mixed_density_matrix(dim, rng=rng, rank=rank, dtype=dtype, device=device)
    if model == "random_pure":
        return random_pure_density_matrix(dim, rng=rng, dtype=dtype, device=device)
    if model == "maximally_mixed":
        return maximally_mixed(dim, dtype=dtype, device=device)

    raise ValueError(
        f"Unsupported state model '{model}'. Supported: "
        f"{{'random_mixed', 'random_pure', 'maximally_mixed'}}."
    )


# ============================================================
# Product states
# ============================================================

def build_product_pure_ket(local_kets: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Build the tensor-product ket from a sequence of local kets.
    """
    if len(local_kets) == 0:
        raise ValueError("local_kets must be a non-empty sequence.")
    normalized = [normalize_state_vector(psi) for psi in local_kets]
    return kron_all(normalized)


def build_product_density(local_states: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Build the tensor-product density matrix from a sequence of local density matrices.
    """
    if len(local_states) == 0:
        raise ValueError("local_states must be a non-empty sequence.")
    for idx, rho in enumerate(local_states):
        if not is_density_matrix(rho):
            raise ValueError(f"local_states[{idx}] is not a valid density matrix.")
    return kron_all(local_states)


def generate_site_density_matrices(
    qubits_per_site: Sequence[int],
    model: str = "random_mixed",
    rng: RNGInput = None,
    rank: Optional[int] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Generate one density matrix per site.
    """
    site_dims = subsystem_dimensions_from_qubits(qubits_per_site)
    dtype = _coerce_complex_dtype(dtype)
    device = _coerce_device(device)

    states: List[torch.Tensor] = []
    for dim in site_dims:
        states.append(
            sample_density_matrix(
                dim,
                model=model,
                rng=rng,
                rank=rank,
                dtype=dtype,
                device=device,
            )
        )
    return tuple(states)


def generate_global_product_state(
    qubits_per_site: Sequence[int],
    site_model: str = "random_mixed",
    rng: RNGInput = None,
    rank: Optional[int] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Generate a global product density matrix over sites.
    """
    local_states = generate_site_density_matrices(
        qubits_per_site=qubits_per_site,
        model=site_model,
        rng=rng,
        rank=rank,
        dtype=dtype,
        device=device,
    )
    return build_product_density(local_states)


# ============================================================
# Global-to-regional reductions
# ============================================================

def reduce_global_state_to_region(
    global_rho: torch.Tensor,
    qubits_per_site: Sequence[int],
    region_sites: Sequence[int],
) -> torch.Tensor:
    """
    Reduce a global site-factorized density operator to a chosen region.
    """
    site_dims = subsystem_dimensions_from_qubits(qubits_per_site)
    keep = tuple(int(s) for s in region_sites)
    return partial_trace(global_rho, dims=site_dims, keep=keep)


def reduce_global_state_to_all_regions(
    global_rho: torch.Tensor,
    cfg: ExperimentConfig,
) -> Dict[str, torch.Tensor]:
    """
    Reduce a global density matrix to all configured regions.
    """
    out: Dict[str, torch.Tensor] = {}
    for region in cfg.regions:
        out[region.name] = reduce_global_state_to_region(
            global_rho=global_rho,
            qubits_per_site=cfg.qubits_per_site,
            region_sites=region.sites,
        )
    return out


def generate_consistent_regional_truth_from_global_product(
    cfg: ExperimentConfig,
    site_model: str = "random_mixed",
    rng: RNGInput = None,
    rank: Optional[int] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    """
    Generate an overlap-consistent family of regional states from a global product state.

    Returns
    -------
    tuple
        (global_rho, site_states, regional_states_dict)
    """
    rank = _default_rank_from_cfg(cfg, rank)

    site_states = generate_site_density_matrices(
        qubits_per_site=cfg.qubits_per_site,
        model=site_model,
        rng=rng,
        rank=rank,
        dtype=cfg.torch_complex_dtype,
        device=cfg.device,
    )
    global_rho = build_product_density(site_states)
    region_states = reduce_global_state_to_all_regions(global_rho, cfg)
    return global_rho, site_states, region_states


# ============================================================
# Region-state initialization
# ============================================================

def initialize_region_state(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
    rng: RNGInput = None,
    method: Optional[str] = None,
    rank: Optional[int] = None,
) -> torch.Tensor:
    """
    Initialize one regional density matrix for optimization.
    """
    region_obj = _region_obj(region, cfg)
    dim = cfg.region_dimension(region_obj)
    chosen_method = region_obj.init_state_method if method is None else str(method)
    rank = _default_rank_from_cfg(cfg, rank)

    rho = sample_density_matrix(
        dim=dim,
        model=chosen_method,
        rng=rng,
        rank=rank,
        dtype=cfg.torch_complex_dtype,
        device=cfg.device,
    )
    return project_to_density_matrix(rho)


def initialize_all_region_states(
    cfg: ExperimentConfig,
    rng: RNGInput = None,
    method_override: Optional[str] = None,
    rank: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Initialize all regional density matrices.
    """
    rank = _default_rank_from_cfg(cfg, rank)

    out: Dict[str, torch.Tensor] = {}
    for region in cfg.regions:
        out[region.name] = initialize_region_state(
            cfg=cfg,
            region=region,
            rng=rng,
            method=method_override,
            rank=rank,
        )
    return out


# ============================================================
# Truth generation from region-local configs
# ============================================================

def generate_independent_regional_truth(
    cfg: ExperimentConfig,
    rng: RNGInput = None,
    rank: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate regional truth states independently from each region's configured model.

    Warning
    -------
    This function does NOT guarantee overlap consistency across regions.
    """
    rank = _default_rank_from_cfg(cfg, rank)

    out: Dict[str, torch.Tensor] = {}
    for region in cfg.regions:
        dim = cfg.region_dimension(region)
        rho = sample_density_matrix(
            dim=dim,
            model=region.true_state_model,
            rng=rng,
            rank=rank,
            dtype=cfg.torch_complex_dtype,
            device=cfg.device,
        )
        out[region.name] = project_to_density_matrix(rho)
    return out


# ============================================================
# Overlap consistency checks
# ============================================================

def overlap_reduction_for_pair(
    cfg: ExperimentConfig,
    region_states: Mapping[str, torch.Tensor],
    region_a: Union[str, RegionConfig],
    region_b: Union[str, RegionConfig],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the two overlap reductions for a pair of regions.
    """
    a = _region_obj(region_a, cfg)
    b = _region_obj(region_b, cfg)

    overlap_sites = cfg.region_overlap_sites(a, b)
    if len(overlap_sites) == 0:
        raise ValueError(
            f"Regions '{a.name}' and '{b.name}' do not overlap, so no overlap reduction exists."
        )

    rho_a = region_states[a.name]
    rho_b = region_states[b.name]

    local_keep_a = [a.sites.index(s) for s in overlap_sites]
    local_keep_b = [b.sites.index(s) for s in overlap_sites]

    dims_a = cfg.region_site_dimensions(a)
    dims_b = cfg.region_site_dimensions(b)

    red_a = partial_trace(rho_a, dims=dims_a, keep=local_keep_a)
    red_b = partial_trace(rho_b, dims=dims_b, keep=local_keep_b)
    return red_a, red_b


def pairwise_overlap_residual(
    cfg: ExperimentConfig,
    region_states: Mapping[str, torch.Tensor],
    region_a: Union[str, RegionConfig],
    region_b: Union[str, RegionConfig],
) -> float:
    """
    Frobenius norm of the overlap mismatch between two regions.
    """
    red_a, red_b = overlap_reduction_for_pair(cfg, region_states, region_a, region_b)
    return frobenius_norm(red_a - red_b)


def all_overlap_residuals(
    cfg: ExperimentConfig,
    region_states: Mapping[str, torch.Tensor],
) -> Dict[Tuple[str, str], float]:
    """
    Compute overlap residuals for all overlapping region pairs.
    """
    out: Dict[Tuple[str, str], float] = {}
    for i, j in cfg.overlap_pairs():
        name_i = cfg.regions[i].name
        name_j = cfg.regions[j].name
        out[(name_i, name_j)] = pairwise_overlap_residual(
            cfg=cfg,
            region_states=region_states,
            region_a=name_i,
            region_b=name_j,
        )
    return out


def are_region_states_overlap_consistent(
    cfg: ExperimentConfig,
    region_states: Mapping[str, torch.Tensor],
    atol: float = 1e-8,
) -> bool:
    """
    Check whether all overlapping region pairs agree on their shared reductions.
    """
    residuals = all_overlap_residuals(cfg, region_states)
    return all(val <= atol for val in residuals.values())


# ============================================================
# Validation helpers for collections of states
# ============================================================

def validate_region_state_collection(
    cfg: ExperimentConfig,
    region_states: Mapping[str, torch.Tensor],
    check_overlap_consistency: bool = False,
    overlap_atol: float = 1e-8,
) -> None:
    """
    Validate a mapping region name -> density matrix.
    """
    expected_names = {region.name for region in cfg.regions}
    provided_names = set(region_states.keys())

    missing = expected_names - provided_names
    extra = provided_names - expected_names

    if missing:
        raise ValueError(f"Missing regional states for regions: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected region names in state collection: {sorted(extra)}.")

    for region in cfg.regions:
        rho = _as_tensor(
            region_states[region.name],
            dtype=cfg.torch_complex_dtype,
            device=cfg.device,
        )
        dim = cfg.region_dimension(region)
        if tuple(rho.shape) != (dim, dim):
            raise ValueError(
                f"State for region '{region.name}' has shape {tuple(rho.shape)}, expected {(dim, dim)}."
            )
        if not is_density_matrix(rho):
            raise ValueError(f"State for region '{region.name}' is not a valid density matrix.")

    if check_overlap_consistency:
        residuals = all_overlap_residuals(cfg, region_states)
        bad = {k: v for k, v in residuals.items() if v > overlap_atol}
        if bad:
            raise ValueError(
                f"Regional states are not overlap-consistent within tolerance {overlap_atol}. "
                f"Residuals: {bad}"
            )


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_random_states() -> None:
    rho_pure = random_pure_density_matrix(dim=4, rng=123)
    rho_mixed = random_mixed_density_matrix(dim=4, rng=456, rank=3)

    assert is_density_matrix(rho_pure)
    assert is_density_matrix(rho_mixed)
    assert torch.isclose(
        torch.real(torch.trace(rho_pure)),
        torch.tensor(1.0, dtype=torch.float64, device=rho_pure.device),
        atol=1e-10,
    )
    assert torch.isclose(
        torch.real(torch.trace(rho_mixed)),
        torch.tensor(1.0, dtype=torch.float64, device=rho_mixed.device),
        atol=1e-10,
    )


def _self_test_product_state() -> None:
    rho0 = random_pure_density_matrix(dim=2, rng=11)
    rho1 = random_mixed_density_matrix(dim=2, rng=22)
    rho = build_product_density([rho0, rho1])

    assert tuple(rho.shape) == (4, 4)
    assert is_density_matrix(rho)

    red0 = partial_trace(rho, dims=[2, 2], keep=[0])
    red1 = partial_trace(rho, dims=[2, 2], keep=[1])

    assert torch.allclose(red0, rho0, atol=1e-10)
    assert torch.allclose(red1, rho1, atol=1e-10)


def _self_test_global_to_regions() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    global_rho, _, region_states = generate_consistent_regional_truth_from_global_product(
        cfg=cfg,
        site_model="random_mixed",
        rng=123,
    )

    assert is_density_matrix(global_rho)
    validate_region_state_collection(cfg, region_states, check_overlap_consistency=True)


def _self_test_initialization() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    init_states = initialize_all_region_states(cfg, rng=7)

    validate_region_state_collection(cfg, init_states, check_overlap_consistency=False)
    assert set(init_states.keys()) == {region.name for region in cfg.regions}


def _self_test_overlap_residuals() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    _, _, region_states = generate_consistent_regional_truth_from_global_product(
        cfg=cfg,
        site_model="random_mixed",
        rng=999,
    )

    residuals = all_overlap_residuals(cfg, region_states)
    assert len(residuals) == 1
    for val in residuals.values():
        assert val <= 1e-10


def _self_test_gpu_smoke() -> None:
    if not torch.cuda.is_available():
        return

    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    cfg.runtime.device = "cuda"

    rho = random_mixed_density_matrix(
        dim=4,
        rng=123,
        rank=3,
        dtype=cfg.torch_complex_dtype,
        device=cfg.device,
    )
    assert rho.device.type == "cuda"
    assert is_density_matrix(rho)

    _, _, region_states = generate_consistent_regional_truth_from_global_product(
        cfg=cfg,
        site_model="random_mixed",
        rng=123,
    )
    for state in region_states.values():
        assert state.device.type == "cuda"


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the states module.
    """
    tests = [
        ("random state generation", _self_test_random_states),
        ("product state construction", _self_test_product_state),
        ("global-to-regional reductions", _self_test_global_to_regions),
        ("regional initialization", _self_test_initialization),
        ("overlap residuals", _self_test_overlap_residuals),
        ("gpu smoke", _self_test_gpu_smoke),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All states self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
