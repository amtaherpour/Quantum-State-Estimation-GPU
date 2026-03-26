from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import torch

from config import ExperimentConfig, RegionConfig
from core_ops import (
    frobenius_norm,
    is_column_stochastic,
    normalize_probability_vector,
    project_to_column_stochastic,
)


# ============================================================
# Numerical defaults
# ============================================================

DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-8


# ============================================================
# Small helpers
# ============================================================

def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _ensure_nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _ensure_probability(value: float, name: str) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1], got {value}.")
    return value


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


def _coerce_rng_seed(rng: Any = None) -> Optional[int]:
    """
    Convert a supported RNG spec into an integer seed when possible.

    Supported forms
    ---------------
    None
    int
    torch.Generator
    """
    if rng is None:
        return None
    if isinstance(rng, int):
        return int(rng)
    if isinstance(rng, torch.Generator):
        return int(rng.initial_seed())
    raise TypeError(
        "rng must be None, an integer seed, or a torch.Generator. "
        f"Got {type(rng)!r}."
    )


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


def _region_obj(cfg: ExperimentConfig, region: Union[RegionConfig, str]) -> RegionConfig:
    if isinstance(region, RegionConfig):
        return region
    return cfg.region_by_name(region)


# ============================================================
# Outcome-count resolution
# ============================================================

def resolve_region_num_outcomes(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
) -> int:
    """
    Resolve the number of measurement outcomes for a region from the config.

    This mirrors the POVM-builder conventions:
    - computational: number of outcomes = region dimension
    - pauli6_single_qubit: 6
    - random_ic: region.povm_num_outcomes if given, else dim^2
    """
    region_obj = _region_obj(cfg, region)
    dim = cfg.region_dimension(region_obj)

    if region_obj.povm_type == "computational":
        return dim
    if region_obj.povm_type == "pauli6_single_qubit":
        return 6
    if region_obj.povm_type == "random_ic":
        return region_obj.povm_num_outcomes if region_obj.povm_num_outcomes is not None else dim * dim

    raise ValueError(
        f"Unsupported povm_type '{region_obj.povm_type}' for region '{region_obj.name}'."
    )


# ============================================================
# Confusion-matrix validation and projection
# ============================================================

def validate_confusion_matrix(
    c: Any,
    num_outcomes: Optional[int] = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """
    Validate a column-stochastic confusion matrix.

    A valid confusion matrix C satisfies:
    - C is square
    - C_{ij} >= 0
    - each column sums to 1
    """
    c_t = _as_torch_tensor(c, dtype=torch.float64 if not isinstance(c, torch.Tensor) else None)
    if c_t.ndim != 2:
        raise ValueError(f"Confusion matrix must be 2D, got shape {tuple(c_t.shape)}.")
    if c_t.shape[0] != c_t.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got shape {tuple(c_t.shape)}.")

    if num_outcomes is not None:
        num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
        if tuple(c_t.shape) != (num_outcomes, num_outcomes):
            raise ValueError(
                f"Confusion matrix has shape {tuple(c_t.shape)}, expected "
                f"{(num_outcomes, num_outcomes)}."
            )

    if torch.any(c_t < -atol):
        min_entry = float(torch.min(c_t).item())
        raise ValueError(
            "Confusion matrix contains negative entries below tolerance. "
            f"Minimum entry = {min_entry:.6e}."
        )

    col_sums = torch.sum(c_t, dim=0)
    ones = torch.ones_like(col_sums)
    if not torch.allclose(col_sums, ones, atol=atol, rtol=rtol):
        raise ValueError(
            "Confusion matrix columns do not sum to 1 within tolerance. "
            f"Column sums = {col_sums}."
        )


def project_confusion_matrix(c: Any) -> torch.Tensor:
    """
    Project a real matrix onto the set of column-stochastic matrices
    by simplex-projecting each column.
    """
    c_t = _as_torch_tensor(c, dtype=torch.float64 if not isinstance(c, torch.Tensor) else None)
    return project_to_column_stochastic(c_t)


def is_valid_confusion_matrix(
    c: Any,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    """
    Return True if the matrix is column-stochastic.
    """
    c_t = _as_torch_tensor(c, dtype=torch.float64 if not isinstance(c, torch.Tensor) else None)
    return is_column_stochastic(c_t, atol=atol, rtol=rtol)


# ============================================================
# Standard confusion-matrix constructors
# ============================================================

def make_identity_confusion(
    num_outcomes: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Identity confusion matrix, corresponding to ideal readout.
    """
    num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
    dtype = _coerce_real_dtype(dtype)
    device = _coerce_device(device)
    return torch.eye(num_outcomes, dtype=dtype, device=device)


def make_uniform_confusion(
    num_outcomes: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Uniform confusion matrix: every recorded outcome is equally likely
    regardless of the ideal outcome.
    """
    num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
    dtype = _coerce_real_dtype(dtype)
    device = _coerce_device(device)
    return torch.full(
        (num_outcomes, num_outcomes),
        1.0 / num_outcomes,
        dtype=dtype,
        device=device,
    )


def make_noisy_identity_confusion(
    num_outcomes: int,
    strength: float = 0.05,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Noisy-identity confusion matrix:
        C = (1 - strength) I + strength U,
    where U is the uniform column-stochastic matrix.
    """
    num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
    strength = _ensure_probability(strength, "strength")
    dtype = _coerce_real_dtype(dtype)
    device = _coerce_device(device)

    eye = make_identity_confusion(num_outcomes, dtype=dtype, device=device)
    uni = make_uniform_confusion(num_outcomes, dtype=dtype, device=device)
    c = (1.0 - strength) * eye + strength * uni
    c = project_confusion_matrix(c)
    validate_confusion_matrix(c, num_outcomes=num_outcomes)
    return c


def make_random_column_stochastic_confusion(
    num_outcomes: int,
    rng: Any = None,
    concentration: float = 1.0,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Sample a random column-stochastic confusion matrix by drawing each column
    independently from a Dirichlet distribution.
    """
    num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
    concentration = float(concentration)
    if concentration <= 0.0:
        raise ValueError(f"concentration must be positive, got {concentration}.")

    dtype = _coerce_real_dtype(dtype)
    device = _coerce_device(device)
    seed = _coerce_rng_seed(rng)

    alpha = torch.full((num_outcomes,), concentration, dtype=dtype, device=device)

    # torch.distributions.Dirichlet does not accept a Generator directly, so we
    # use a temporary forked RNG context when a seed is supplied.
    if seed is None:
        samples = torch.distributions.Dirichlet(alpha).sample((num_outcomes,))
    else:
        devices = []
        if device.type == "cuda" and torch.cuda.is_available():
            devices = [device]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            samples = torch.distributions.Dirichlet(alpha).sample((num_outcomes,))

    c = samples.transpose(0, 1).contiguous()
    validate_confusion_matrix(c, num_outcomes=num_outcomes)
    return c


# ============================================================
# Applying readout noise
# ============================================================

def apply_confusion_matrix(
    c: Any,
    ideal_probabilities: Any,
    prob_floor: float = 0.0,
) -> torch.Tensor:
    """
    Apply a confusion matrix to an ideal probability vector:
        p_noisy = C p_ideal.
    """
    c_t = _as_torch_tensor(c, dtype=torch.float64 if not isinstance(c, torch.Tensor) else None)
    validate_confusion_matrix(c_t)

    p = _as_torch_tensor(
        ideal_probabilities,
        dtype=c_t.dtype,
        device=c_t.device,
    ).reshape(-1)

    if p.numel() != c_t.shape[1]:
        raise ValueError(
            f"ideal_probabilities has length {p.numel()}, but confusion matrix expects "
            f"length {c_t.shape[1]}."
        )

    p = normalize_probability_vector(p, floor=0.0)
    q = c_t @ p
    q = torch.real(q)

    prob_floor = _ensure_nonnegative_float(prob_floor, "prob_floor")
    if prob_floor > 0.0:
        q = torch.clamp(q, min=prob_floor)
        q = q / torch.sum(q)
    else:
        q = normalize_probability_vector(q, floor=0.0)

    return q


def apply_confusion_to_region_probabilities(
    region_probabilities: Mapping[str, Any],
    region_confusions: Mapping[str, Any],
    prob_floor: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Apply region-specific confusion matrices to region-specific probability vectors.
    """
    prob_names = set(region_probabilities.keys())
    conf_names = set(region_confusions.keys())

    missing = prob_names - conf_names
    extra = conf_names - prob_names

    if missing:
        raise ValueError(f"Missing confusion matrices for regions: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected confusion matrices for regions: {sorted(extra)}.")

    out: Dict[str, torch.Tensor] = {}
    for name, p in region_probabilities.items():
        out[name] = apply_confusion_matrix(
            region_confusions[name],
            p,
            prob_floor=prob_floor,
        )
    return out


# ============================================================
# Regularization helpers
# ============================================================

def confusion_frobenius_regularizer(
    c: Any,
    reference: Any,
) -> float:
    """
    Squared Frobenius regularizer:
        ||C - C_ref||_F^2
    """
    c_t = _as_torch_tensor(c, dtype=torch.float64 if not isinstance(c, torch.Tensor) else None)
    reference_t = _as_torch_tensor(
        reference,
        dtype=c_t.dtype,
        device=c_t.device,
    )

    if tuple(c_t.shape) != tuple(reference_t.shape):
        raise ValueError(
            f"c and reference must have the same shape, got {tuple(c_t.shape)} "
            f"and {tuple(reference_t.shape)}."
        )

    diff = c_t - reference_t
    return float(torch.sum(diff * diff).item())


def confusion_identity_distance(c: Any) -> float:
    """
    Frobenius distance from the identity confusion matrix.
    """
    c_t = _as_torch_tensor(c, dtype=torch.float64 if not isinstance(c, torch.Tensor) else None)
    validate_confusion_matrix(c_t)
    eye = torch.eye(c_t.shape[0], dtype=c_t.dtype, device=c_t.device)
    return frobenius_norm(c_t - eye)


# ============================================================
# Config-based builders
# ============================================================

def build_true_region_confusion(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
    rng: Any = None,
    concentration: float = 1.0,
) -> torch.Tensor:
    """
    Build the ground-truth confusion matrix for one region using
    region.true_confusion_model.
    """
    region_obj = _region_obj(cfg, region)
    m = resolve_region_num_outcomes(cfg, region_obj)
    model = region_obj.true_confusion_model

    kwargs = {"dtype": cfg.torch_real_dtype, "device": cfg.device}

    if model == "identity":
        return make_identity_confusion(m, **kwargs)

    if model == "noisy_identity":
        return make_noisy_identity_confusion(
            num_outcomes=m,
            strength=region_obj.confusion_strength,
            **kwargs,
        )

    if model == "random_column_stochastic":
        return make_random_column_stochastic_confusion(
            num_outcomes=m,
            rng=rng,
            concentration=concentration,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported true_confusion_model '{model}' for region '{region_obj.name}'."
    )


def build_initial_region_confusion(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
    rng: Any = None,
) -> torch.Tensor:
    """
    Build the initial confusion matrix for one region using
    region.init_confusion_method.
    """
    region_obj = _region_obj(cfg, region)
    m = resolve_region_num_outcomes(cfg, region_obj)
    method = region_obj.init_confusion_method

    kwargs = {"dtype": cfg.torch_real_dtype, "device": cfg.device}

    if method == "identity":
        return make_identity_confusion(m, **kwargs)

    if method == "uniform":
        return make_uniform_confusion(m, **kwargs)

    if method == "noisy_identity":
        return make_noisy_identity_confusion(
            num_outcomes=m,
            strength=region_obj.confusion_strength,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported init_confusion_method '{method}' for region '{region_obj.name}'."
    )


def build_reference_region_confusion(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
) -> torch.Tensor:
    """
    Build the reference confusion matrix used in regularization.

    At the current config stage, only 'identity' is supported.
    """
    region_obj = _region_obj(cfg, region)
    m = resolve_region_num_outcomes(cfg, region_obj)

    if region_obj.reference_confusion_type == "identity":
        return make_identity_confusion(
            m,
            dtype=cfg.torch_real_dtype,
            device=cfg.device,
        )

    raise ValueError(
        f"Unsupported reference_confusion_type '{region_obj.reference_confusion_type}' "
        f"for region '{region_obj.name}'."
    )


def build_all_true_confusions(
    cfg: ExperimentConfig,
    rng: Any = None,
    concentration: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Build ground-truth confusion matrices for all regions.
    """
    out: Dict[str, torch.Tensor] = {}
    for region in cfg.regions:
        out[region.name] = build_true_region_confusion(
            cfg=cfg,
            region=region,
            rng=rng,
            concentration=concentration,
        )
    return out


def build_all_initial_confusions(
    cfg: ExperimentConfig,
    rng: Any = None,
) -> Dict[str, torch.Tensor]:
    """
    Build initial confusion matrices for all regions.
    """
    out: Dict[str, torch.Tensor] = {}
    for region in cfg.regions:
        out[region.name] = build_initial_region_confusion(
            cfg=cfg,
            region=region,
            rng=rng,
        )
    return out


def build_all_reference_confusions(
    cfg: ExperimentConfig,
) -> Dict[str, torch.Tensor]:
    """
    Build reference confusion matrices for all regions.
    """
    out: Dict[str, torch.Tensor] = {}
    for region in cfg.regions:
        out[region.name] = build_reference_region_confusion(cfg, region)
    return out


# ============================================================
# Validation for collections
# ============================================================

def validate_region_confusion_collection(
    cfg: ExperimentConfig,
    confusions: Mapping[str, Any],
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """
    Validate a region-name -> confusion-matrix mapping against the config.
    """
    expected_names = {region.name for region in cfg.regions}
    provided_names = set(confusions.keys())

    missing = expected_names - provided_names
    extra = provided_names - expected_names

    if missing:
        raise ValueError(f"Missing confusion matrices for regions: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected region names in confusion collection: {sorted(extra)}.")

    for region in cfg.regions:
        m = resolve_region_num_outcomes(cfg, region)
        validate_confusion_matrix(
            confusions[region.name],
            num_outcomes=m,
            atol=atol,
            rtol=rtol,
        )


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_basic_constructors() -> None:
    m = 5
    eye = make_identity_confusion(m)
    uni = make_uniform_confusion(m)
    noisy = make_noisy_identity_confusion(m, strength=0.2)

    validate_confusion_matrix(eye, num_outcomes=m)
    validate_confusion_matrix(uni, num_outcomes=m)
    validate_confusion_matrix(noisy, num_outcomes=m)

    assert torch.allclose(eye, torch.eye(m, dtype=eye.dtype))
    assert torch.allclose(torch.sum(uni, dim=0), torch.ones(m, dtype=uni.dtype))
    assert torch.allclose(torch.sum(noisy, dim=0), torch.ones(m, dtype=noisy.dtype))


def _self_test_random_column_stochastic() -> None:
    m = 7
    c = make_random_column_stochastic_confusion(m, rng=123, concentration=0.7)
    validate_confusion_matrix(c, num_outcomes=m)
    assert tuple(c.shape) == (m, m)
    assert bool(torch.all(c >= 0.0))
    assert torch.allclose(torch.sum(c, dim=0), torch.ones(m, dtype=c.dtype), atol=1e-10)


def _self_test_apply_confusion() -> None:
    c = make_noisy_identity_confusion(3, strength=0.3)
    p = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    q = apply_confusion_matrix(c, p)

    assert tuple(q.shape) == (3,)
    assert torch.isclose(torch.sum(q), torch.tensor(1.0, dtype=q.dtype), atol=1e-10)
    assert bool(torch.all(q >= 0.0))
    assert torch.allclose(q, c[:, 1], atol=1e-10)


def _self_test_regularizer() -> None:
    c = make_noisy_identity_confusion(4, strength=0.1)
    ref = make_identity_confusion(4)

    val = confusion_frobenius_regularizer(c, ref)
    dist = confusion_identity_distance(c)

    assert val >= 0.0
    assert abs(val - dist ** 2) <= 1e-10


def _self_test_config_builders() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()

    true_confusions = build_all_true_confusions(cfg, rng=2024)
    init_confusions = build_all_initial_confusions(cfg, rng=2024)
    ref_confusions = build_all_reference_confusions(cfg)

    validate_region_confusion_collection(cfg, true_confusions)
    validate_region_confusion_collection(cfg, init_confusions)
    validate_region_confusion_collection(cfg, ref_confusions)

    assert set(true_confusions.keys()) == {region.name for region in cfg.regions}
    assert set(init_confusions.keys()) == {region.name for region in cfg.regions}
    assert set(ref_confusions.keys()) == {region.name for region in cfg.regions}

    for region in cfg.regions:
        m = resolve_region_num_outcomes(cfg, region)
        assert tuple(true_confusions[region.name].shape) == (m, m)
        assert tuple(init_confusions[region.name].shape) == (m, m)
        assert tuple(ref_confusions[region.name].shape) == (m, m)
        assert true_confusions[region.name].device == cfg.device
        assert init_confusions[region.name].device == cfg.device
        assert ref_confusions[region.name].device == cfg.device


def _self_test_gpu_smoke() -> None:
    if not torch.cuda.is_available():
        return

    c = make_random_column_stochastic_confusion(
        5,
        rng=321,
        concentration=1.2,
        dtype=torch.float64,
        device="cuda",
    )
    p = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.25], dtype=torch.float64, device="cuda")
    q = apply_confusion_matrix(c, p)

    assert c.device.type == "cuda"
    assert q.device.type == "cuda"
    validate_confusion_matrix(c, num_outcomes=5)
    assert torch.isclose(torch.sum(q), torch.tensor(1.0, dtype=q.dtype, device=q.device), atol=1e-10)


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the noise module.
    """
    tests = [
        ("basic constructors", _self_test_basic_constructors),
        ("random column stochastic", _self_test_random_column_stochastic),
        ("apply confusion", _self_test_apply_confusion),
        ("regularizer", _self_test_regularizer),
        ("config builders", _self_test_config_builders),
        ("gpu smoke", _self_test_gpu_smoke),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All noise self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
