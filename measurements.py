from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from config import ExperimentConfig, RegionConfig
from core_ops import (
    dagger,
    frobenius_norm,
    hermitian_part,
    identity,
    is_psd,
)


# ============================================================
# Numerical defaults
# ============================================================

DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-8
DEFAULT_PROB_FLOOR = 1e-12


# ============================================================
# Small internal helpers
# ============================================================

def _real_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise ValueError(f"Unsupported complex dtype: {dtype}.")


def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _check_square_matrix(a: torch.Tensor, name: str = "matrix") -> None:
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor, got shape {tuple(a.shape)}.")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be square, got shape {tuple(a.shape)}.")


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


def _coerce_torch_generator(
    rng: Any = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> Optional[torch.Generator]:
    """
    Accept None, int seed, or torch.Generator.
    """
    if rng is None:
        return None
    if isinstance(rng, torch.Generator):
        return rng

    target = _coerce_device(device)
    gen = torch.Generator(device=target.type)
    gen.manual_seed(int(rng))
    return gen


def _as_torch_tensor(
    x: Any,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        out = x
        if dtype is not None or device is not None:
            out = out.to(
                dtype=out.dtype if dtype is None else dtype,
                device=out.device if device is None else device,
            )
        return out
    return torch.as_tensor(x, dtype=dtype, device=device)


def _as_complex_matrix(
    x: Any,
    *,
    dtype: torch.dtype = torch.complex128,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    out = _as_torch_tensor(x, dtype=dtype, device=device)
    _check_square_matrix(out, "matrix")
    return out


def _coerce_region_obj(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
) -> RegionConfig:
    return cfg.region_by_name(region) if isinstance(region, str) else region


def _ket_to_density(psi: Any, *, dtype: Optional[torch.dtype] = None,
                    device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    psi = _as_torch_tensor(
        psi,
        dtype=_coerce_complex_dtype(dtype),
        device=device,
    ).reshape(-1)
    norm = torch.linalg.vector_norm(psi)
    if float(norm.item()) <= 1e-14:
        raise ValueError("Cannot build a density matrix from a numerically zero ket.")
    psi = psi / norm
    return torch.outer(psi, psi.conj())


def _projector(psi: Any, *, dtype: Optional[torch.dtype] = None,
               device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    return _ket_to_density(psi, dtype=dtype, device=device)


def _eigh_inverse_sqrt_psd(a: Any, atol: float = 1e-12) -> torch.Tensor:
    """
    Return A^{-1/2} for a Hermitian positive definite matrix A.
    """
    a_t = hermitian_part(_as_torch_tensor(a))
    _check_square_matrix(a_t, "a")

    evals, evecs = torch.linalg.eigh(a_t)
    min_eval = float(torch.min(evals).item())
    if min_eval <= atol:
        raise ValueError(
            "Cannot compute inverse square root because the matrix is numerically singular "
            f"or not strictly positive definite. Minimum eigenvalue = {min_eval}."
        )

    inv_sqrt_evals = torch.rsqrt(evals.real)
    return (evecs * inv_sqrt_evals.unsqueeze(0)) @ dagger(evecs)


def _effects_to_tuple(
    effects: Sequence[Any],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, ...]:
    if len(effects) == 0:
        raise ValueError("effects must be a non-empty sequence.")

    out: List[torch.Tensor] = []
    chosen_dtype: Optional[torch.dtype] = _coerce_complex_dtype(dtype) if dtype is not None else None
    chosen_device: Optional[torch.device] = _coerce_device(device) if device is not None else None

    for idx, e in enumerate(effects):
        if isinstance(e, torch.Tensor):
            if chosen_dtype is None:
                chosen_dtype = e.dtype if e.is_complex() else torch.complex128
            if chosen_device is None:
                chosen_device = e.device
            break

    if chosen_dtype is None:
        chosen_dtype = torch.complex128
    if chosen_device is None:
        chosen_device = torch.device("cpu")

    for idx, e in enumerate(effects):
        e_t = _as_torch_tensor(e, dtype=chosen_dtype, device=chosen_device)
        _check_square_matrix(e_t, f"effects[{idx}]")
        out.append(e_t)

    dim = out[0].shape[0]
    for idx, e_t in enumerate(out):
        if e_t.shape != (dim, dim):
            raise ValueError(
                "All POVM effects must have the same shape. "
                f"effects[0] has shape {(dim, dim)} but effects[{idx}] has shape {tuple(e_t.shape)}."
            )
    return tuple(out)


# ============================================================
# POVM data structure
# ============================================================

@dataclass(frozen=True)
class POVM:
    """
    Positive operator-valued measure (POVM).
    """
    name: str
    effects: Tuple[torch.Tensor, ...]
    dim: int
    num_outcomes: int
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        effects = _effects_to_tuple(self.effects)
        object.__setattr__(self, "effects", effects)

        dim = _ensure_positive_int(self.dim, "dim")
        num_outcomes = _ensure_positive_int(self.num_outcomes, "num_outcomes")

        if len(effects) != num_outcomes:
            raise ValueError(
                f"num_outcomes={num_outcomes} but {len(effects)} effects were provided."
            )
        if any(e.shape != (dim, dim) for e in effects):
            raise ValueError(f"All effects must have shape {(dim, dim)}.")

        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "num_outcomes", num_outcomes)

    @property
    def device(self) -> torch.device:
        return self.effects[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.effects[0].dtype

    def stacked_effects(self) -> torch.Tensor:
        return torch.stack(self.effects, dim=0)

    def to(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "POVM":
        new_effects = tuple(
            e.to(
                dtype=e.dtype if dtype is None else dtype,
                device=e.device if device is None else device,
            )
            for e in self.effects
        )
        return POVM(
            name=self.name,
            effects=new_effects,
            dim=self.dim,
            num_outcomes=self.num_outcomes,
            metadata=None if self.metadata is None else dict(self.metadata),
        )

    def validate(
        self,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        check_psd: bool = True,
    ) -> None:
        validate_povm(
            self.effects,
            dim=self.dim,
            atol=atol,
            rtol=rtol,
            check_psd=check_psd,
        )


# ============================================================
# POVM validation
# ============================================================

def povm_identity_residual(
    effects: Sequence[Any],
    dim: Optional[int] = None,
) -> float:
    effects_t = _effects_to_tuple(effects)
    if dim is None:
        dim = effects_t[0].shape[0]
    dim = int(dim)

    s = torch.zeros(
        (dim, dim),
        dtype=effects_t[0].dtype,
        device=effects_t[0].device,
    )
    for e in effects_t:
        s = s + e
    return frobenius_norm(s - identity(dim, dtype=effects_t[0].dtype, device=effects_t[0].device))


def validate_povm(
    effects: Sequence[Any],
    dim: Optional[int] = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    check_psd: bool = True,
) -> None:
    r"""
    Validate the POVM constraints:
        E_m \succeq 0,    \sum_m E_m = I.
    """
    effects_t = _effects_to_tuple(effects)
    if dim is None:
        dim = effects_t[0].shape[0]
    dim = _ensure_positive_int(dim, "dim")

    for idx, e in enumerate(effects_t):
        if e.shape != (dim, dim):
            raise ValueError(
                f"effects[{idx}] has shape {tuple(e.shape)}, expected {(dim, dim)}."
            )
        if not torch.allclose(e, dagger(e), atol=atol, rtol=rtol):
            raise ValueError(f"effects[{idx}] is not Hermitian within tolerance.")
        if check_psd and not is_psd(e, atol=atol):
            raise ValueError(f"effects[{idx}] is not PSD within tolerance.")

    s = torch.zeros(
        (dim, dim),
        dtype=effects_t[0].dtype,
        device=effects_t[0].device,
    )
    for e in effects_t:
        s = s + e

    ident = identity(dim, dtype=effects_t[0].dtype, device=effects_t[0].device)
    if not torch.allclose(s, ident, atol=atol, rtol=rtol):
        residual = frobenius_norm(s - ident)
        raise ValueError(
            f"POVM effects do not sum to the identity within tolerance. Residual = {residual:.6e}."
        )


# ============================================================
# Standard POVM constructors
# ============================================================

def make_computational_povm(
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> POVM:
    dim = _ensure_positive_int(dim, "dim")
    dtype = _coerce_complex_dtype(dtype)
    device = _coerce_device(device)

    effects: List[torch.Tensor] = []
    for k in range(dim):
        e = torch.zeros((dim, dim), dtype=dtype, device=device)
        e[k, k] = 1.0
        effects.append(e)

    povm = POVM(
        name=f"computational_dim_{dim}",
        effects=tuple(effects),
        dim=dim,
        num_outcomes=dim,
        metadata={"type": "computational"},
    )
    povm.validate()
    return povm


def make_pauli6_single_qubit_povm(
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> POVM:
    r"""
    Construct the 6-outcome single-qubit Pauli POVM.
    """
    dtype = _coerce_complex_dtype(dtype)
    device = _coerce_device(device)
    real_dtype = _real_dtype_for_complex(dtype)

    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=real_dtype, device=device))

    zero = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
    one = torch.tensor([0.0, 1.0], dtype=dtype, device=device)
    plus = torch.tensor([1.0, 1.0], dtype=dtype, device=device) / sqrt2
    minus = torch.tensor([1.0, -1.0], dtype=dtype, device=device) / sqrt2
    plus_i = torch.tensor([1.0, 1.0j], dtype=dtype, device=device) / sqrt2
    minus_i = torch.tensor([1.0, -1.0j], dtype=dtype, device=device) / sqrt2

    effects = tuple(
        (1.0 / 3.0) * _projector(psi, dtype=dtype, device=device)
        for psi in (zero, one, plus, minus, plus_i, minus_i)
    )

    povm = POVM(
        name="pauli6_single_qubit",
        effects=effects,
        dim=2,
        num_outcomes=6,
        metadata={"type": "pauli6_single_qubit"},
    )
    povm.validate()
    return povm


def make_random_ic_povm(
    dim: int,
    num_outcomes: Optional[int] = None,
    rng: Any = None,
    max_tries: int = 20,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> POVM:
    r"""
    Construct a random informationally complete rank-1 POVM.
    """
    dim = _ensure_positive_int(dim, "dim")
    if num_outcomes is None:
        num_outcomes = dim * dim
    num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
    if num_outcomes < dim * dim:
        raise ValueError(
            f"Random informationally complete POVM requires num_outcomes >= dim^2 = {dim * dim}, "
            f"got {num_outcomes}."
        )

    dtype = _coerce_complex_dtype(dtype)
    device = _coerce_device(device)
    real_dtype = _real_dtype_for_complex(dtype)
    gen = _coerce_torch_generator(rng, device=device)

    for attempt in range(max_tries):
        vectors: List[torch.Tensor] = []
        frame = torch.zeros((dim, dim), dtype=dtype, device=device)

        for _ in range(num_outcomes):
            v_real = torch.randn(dim, dtype=real_dtype, device=device, generator=gen)
            v_imag = torch.randn(dim, dtype=real_dtype, device=device, generator=gen)
            v = v_real + 1j * v_imag
            f = _projector(v, dtype=dtype, device=device)
            vectors.append(f)
            frame = frame + f

        try:
            frame_inv_sqrt = _eigh_inverse_sqrt_psd(frame, atol=1e-12)
        except ValueError:
            continue

        effects = tuple(
            hermitian_part(frame_inv_sqrt @ f @ frame_inv_sqrt)
            for f in vectors
        )

        povm = POVM(
            name=f"random_ic_dim_{dim}_M_{num_outcomes}",
            effects=effects,
            dim=dim,
            num_outcomes=num_outcomes,
            metadata={
                "type": "random_ic",
                "attempt": attempt + 1,
            },
        )
        povm.validate(atol=1e-8, rtol=1e-7)
        return povm

    raise RuntimeError(
        f"Failed to construct a numerically stable random IC POVM after {max_tries} attempts."
    )


# ============================================================
# Measurement maps and Born-rule probabilities
# ============================================================

def born_probability_vector(
    rho: Any,
    effects: Sequence[Any],
    prob_floor: float = 0.0,
) -> torch.Tensor:
    r"""
    Compute the Born probability vector
        p_m = Tr(E_m rho).
    """
    effects_t = _effects_to_tuple(effects)
    dim = effects_t[0].shape[0]
    rho_t = _as_torch_tensor(
        rho,
        dtype=effects_t[0].dtype,
        device=effects_t[0].device,
    )
    _check_square_matrix(rho_t, "rho")

    if rho_t.shape != (dim, dim):
        raise ValueError(
            f"rho has shape {tuple(rho_t.shape)}, but effects act on dimension {dim}."
        )

    eff_stack = torch.stack(effects_t, dim=0)
    probs = torch.real(torch.einsum("mij,ji->m", eff_stack, rho_t))

    tiny = torch.finfo(probs.dtype).eps * 10.0
    probs = torch.where(torch.abs(probs) < tiny, torch.zeros_like(probs), probs)

    if prob_floor < 0.0:
        raise ValueError(f"prob_floor must be nonnegative, got {prob_floor}.")
    if prob_floor > 0.0:
        probs = torch.clamp(probs, min=float(prob_floor))
        s = torch.sum(probs)
        if float(s.item()) <= 0.0:
            raise ValueError("Probability vector has non-positive total mass after flooring.")
        probs = probs / s

    return probs


def measurement_map(
    rho: Any,
    povm: Union[POVM, Sequence[Any]],
) -> torch.Tensor:
    """
    Apply the linear POVM measurement map to an operator rho.
    """
    effects = povm.effects if isinstance(povm, POVM) else povm
    return born_probability_vector(rho, effects, prob_floor=0.0)


def measurement_map_adjoint(
    weights: Any,
    povm: Union[POVM, Sequence[Any]],
) -> torch.Tensor:
    r"""
    Apply the adjoint measurement map:
        M^*(w) = sum_m w_m E_m.
    """
    effects = povm.effects if isinstance(povm, POVM) else _effects_to_tuple(povm)
    weights_t = _as_torch_tensor(
        weights,
        dtype=_real_dtype_for_complex(effects[0].dtype),
        device=effects[0].device,
    ).reshape(-1)

    if len(weights_t) != len(effects):
        raise ValueError(
            f"weights has length {len(weights_t)} but POVM has {len(effects)} outcomes."
        )

    eff_stack = torch.stack(effects, dim=0)
    out = torch.einsum("m,mij->ij", weights_t, eff_stack)
    return hermitian_part(out)


def expected_counts(
    rho: Any,
    povm: Union[POVM, Sequence[Any]],
    shots: int,
) -> torch.Tensor:
    shots = _ensure_positive_int(shots, "shots")
    probs = measurement_map(rho, povm)
    return shots * probs


def povm_effect_traces(
    povm: Union[POVM, Sequence[Any]],
) -> torch.Tensor:
    effects = povm.effects if isinstance(povm, POVM) else _effects_to_tuple(povm)
    return torch.tensor(
        [float(torch.real(torch.trace(e)).item()) for e in effects],
        dtype=_real_dtype_for_complex(effects[0].dtype),
        device=effects[0].device,
    )


# ============================================================
# Config-based builders
# ============================================================

def build_region_povm(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
    rng: Any = None,
) -> POVM:
    region_obj = _coerce_region_obj(cfg, region)
    dim = cfg.region_dimension(region_obj)
    povm_type = region_obj.povm_type
    num_outcomes = region_obj.povm_num_outcomes

    dtype = cfg.torch_complex_dtype
    device = cfg.device

    if povm_type == "computational":
        povm = make_computational_povm(dim, dtype=dtype, device=device)
        if num_outcomes is not None and num_outcomes != povm.num_outcomes:
            raise ValueError(
                f"Region '{region_obj.name}' requested povm_num_outcomes={num_outcomes}, "
                f"but the computational POVM has exactly {povm.num_outcomes} outcomes."
            )
        return povm

    if povm_type == "pauli6_single_qubit":
        if dim != 2:
            raise ValueError(
                f"Region '{region_obj.name}' uses pauli6_single_qubit but has dimension {dim}."
            )
        povm = make_pauli6_single_qubit_povm(dtype=dtype, device=device)
        if num_outcomes is not None and num_outcomes != povm.num_outcomes:
            raise ValueError(
                f"Region '{region_obj.name}' requested povm_num_outcomes={num_outcomes}, "
                f"but pauli6_single_qubit has exactly 6 outcomes."
            )
        return povm

    if povm_type == "random_ic":
        if num_outcomes is None:
            num_outcomes = dim * dim
        return make_random_ic_povm(
            dim=dim,
            num_outcomes=num_outcomes,
            rng=rng,
            dtype=dtype,
            device=device,
        )

    raise ValueError(
        f"Unsupported povm_type '{povm_type}' for region '{region_obj.name}'."
    )


def build_all_region_povms(
    cfg: ExperimentConfig,
    rng: Any = None,
) -> Dict[str, POVM]:
    """
    Build POVMs for all regions in the experiment configuration.
    """
    if rng is None:
        rng = cfg.make_torch_generator()
    else:
        rng = _coerce_torch_generator(rng, device=cfg.device)

    out: Dict[str, POVM] = {}
    for region in cfg.regions:
        out[region.name] = build_region_povm(cfg, region, rng=rng)
    return out


# ============================================================
# Validation of POVM collections
# ============================================================

def validate_region_povm_collection(
    cfg: ExperimentConfig,
    povms: Dict[str, POVM],
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """
    Validate a region-name -> POVM mapping against the experiment config.
    """
    expected_names = {region.name for region in cfg.regions}
    provided_names = set(povms.keys())

    missing = expected_names - provided_names
    extra = provided_names - expected_names

    if missing:
        raise ValueError(f"Missing POVMs for regions: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected region names in POVM collection: {sorted(extra)}.")

    for region in cfg.regions:
        povm = povms[region.name]
        expected_dim = cfg.region_dimension(region)

        if povm.dim != expected_dim:
            raise ValueError(
                f"POVM for region '{region.name}' has dim={povm.dim}, expected {expected_dim}."
            )
        povm.validate(atol=atol, rtol=rtol, check_psd=True)


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_computational_povm() -> None:
    povm = make_computational_povm(4)
    povm.validate()

    rho = torch.zeros((4, 4), dtype=torch.complex128)
    rho[2, 2] = 1.0

    p = measurement_map(rho, povm)
    target = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)

    assert torch.allclose(p, target, atol=1e-10)
    assert torch.isclose(torch.sum(p), torch.tensor(1.0, dtype=p.dtype), atol=1e-10)


def _self_test_pauli6_povm() -> None:
    povm = make_pauli6_single_qubit_povm()
    povm.validate()

    rho0 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128)
    p = measurement_map(rho0, povm)

    assert len(p) == 6
    assert torch.isclose(torch.sum(p), torch.tensor(1.0, dtype=p.dtype), atol=1e-10)

    target = torch.tensor([1 / 3, 0.0, 1 / 6, 1 / 6, 1 / 6, 1 / 6], dtype=torch.float64)
    assert torch.allclose(p, target, atol=1e-10)


def _self_test_random_ic_povm() -> None:
    povm = make_random_ic_povm(dim=3, num_outcomes=9, rng=123)
    povm.validate()

    residual = povm_identity_residual(povm.effects, dim=3)
    assert residual <= 1e-8

    rho = torch.eye(3, dtype=torch.complex128) / 3.0
    p = measurement_map(rho, povm)
    assert torch.isclose(torch.sum(p), torch.tensor(1.0, dtype=p.dtype), atol=1e-8)
    assert bool(torch.all(p >= -1e-10))


def _self_test_measurement_adjoint() -> None:
    povm = make_random_ic_povm(dim=2, num_outcomes=4, rng=7)
    rho = torch.tensor(
        [[0.7, 0.1 - 0.05j], [0.1 + 0.05j, 0.3]],
        dtype=torch.complex128,
    )
    rho = hermitian_part(rho)
    rho = rho / torch.trace(rho)

    w = torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=torch.float64)
    lhs = float(torch.dot(w, measurement_map(rho, povm)).item())
    rhs = float(torch.real(torch.trace(measurement_map_adjoint(w, povm) @ rho)).item())

    assert abs(lhs - rhs) <= 1e-10


def _self_test_config_builders() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    povms = build_all_region_povms(cfg, rng=2024)
    validate_region_povm_collection(cfg, povms)

    assert set(povms.keys()) == {region.name for region in cfg.regions}
    for region in cfg.regions:
        assert povms[region.name].dim == cfg.region_dimension(region)
        assert povms[region.name].device == cfg.device


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the measurements module.
    """
    tests = [
        ("computational POVM", _self_test_computational_povm),
        ("pauli-6 single-qubit POVM", _self_test_pauli6_povm),
        ("random IC POVM", _self_test_random_ic_povm),
        ("measurement adjoint identity", _self_test_measurement_adjoint),
        ("config-based POVM builders", _self_test_config_builders),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All measurements self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
