# core_ops.py
from __future__ import annotations

import importlib
import importlib.util
from typing import List, Optional, Sequence

import numpy as np


# Optional CuPy backend (if installed). No hard dependency.
_cp_spec = importlib.util.find_spec("cupy")
cp = importlib.import_module("cupy") if _cp_spec is not None else None


# ============================================================
# Numerical defaults
# ============================================================

DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-8
DEFAULT_PROB_FLOOR = 1e-12


# ============================================================
# Backend helpers
# ============================================================

def _is_cupy_array(x) -> bool:
    return cp is not None and isinstance(x, cp.ndarray)


def _get_array_module(*objs):
    """Return `cupy` if any object is a CuPy array, else `numpy`."""
    for obj in objs:
        if _is_cupy_array(obj):
            return cp
        if isinstance(obj, (list, tuple)):
            for y in obj:
                if _is_cupy_array(y):
                    return cp
    return np


def _to_scalar(x):
    if hasattr(x, "item"):
        return x.item()
    return x


def _as_array(x, dtype=None, xp=None):
    xp = np if xp is None else xp
    return xp.asarray(x, dtype=dtype)


def _as_numpy_array(x, dtype=None) -> np.ndarray:
    """Convert input to a NumPy array (host array)."""
    if _is_cupy_array(x):
        return cp.asnumpy(x).astype(dtype) if dtype is not None else cp.asnumpy(x)
    return np.asarray(x, dtype=dtype)


def _check_square_matrix(a, name: str = "matrix") -> None:
    """Raise ValueError if `a` is not a square 2D array."""
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {a.shape}.")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be square, got shape {a.shape}.")


def _check_same_shape(a, b, name_a: str = "a", name_b: str = "b") -> None:
    """Raise ValueError if two arrays do not have the same shape."""
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape, got {a.shape} and {b.shape}."
        )


def _prod_int(values: Sequence[int]) -> int:
    """Integer product of a sequence of positive integers."""
    out = 1
    for v in values:
        out *= int(v)
    return out


def _validate_dims(dims: Sequence[int], name: str = "dims") -> List[int]:
    """Validate subsystem dimensions and return them as a list of ints."""
    if len(dims) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of positive integers.")
    dims_list = [int(d) for d in dims]
    if any(d <= 0 for d in dims_list):
        raise ValueError(f"All entries of {name} must be positive. Got {dims_list}.")
    return dims_list


def _validate_indices(indices: Sequence[int], n: int, name: str = "indices") -> List[int]:
    """Validate that indices are unique integers in range [0, n)."""
    idx = [int(i) for i in indices]
    if len(set(idx)) != len(idx):
        raise ValueError(f"{name} must not contain duplicates. Got {idx}.")
    if any(i < 0 or i >= n for i in idx):
        raise ValueError(f"{name} must lie in [0, {n - 1}]. Got {idx}.")
    return idx


# ============================================================
# Complex / Hermitian helpers
# ============================================================

def dagger(a):
    """Conjugate transpose of a matrix."""
    xp = _get_array_module(a)
    a = _as_array(a, xp=xp)
    return xp.conjugate(a.T)


def hermitian_part(a):
    r"""Return the Hermitian part: (A + A^\dagger) / 2."""
    xp = _get_array_module(a)
    a = _as_array(a, dtype=xp.complex128, xp=xp)
    _check_square_matrix(a, "a")
    return 0.5 * (a + dagger(a))


def antihermitian_part(a):
    r"""Return the anti-Hermitian part: (A - A^\dagger) / 2."""
    xp = _get_array_module(a)
    a = _as_array(a, dtype=xp.complex128, xp=xp)
    _check_square_matrix(a, "a")
    return 0.5 * (a - dagger(a))


def real_if_close_scalar(x, atol: float = DEFAULT_ATOL) -> float | complex:
    """Return real part if imaginary part is numerically negligible."""
    xp = _get_array_module(x)
    x = _to_scalar(_as_array(x, xp=xp).reshape(()))
    if abs(float(np.imag(x))) <= atol:
        return float(np.real(x))
    return x


def real_if_close_array(x, atol: float = DEFAULT_ATOL):
    """Return a real array if all imaginary parts are numerically negligible."""
    xp = _get_array_module(x)
    x = _as_array(x, xp=xp)
    imag_max = _to_scalar(xp.max(xp.abs(xp.imag(x)))) if x.size > 0 else 0.0
    if float(imag_max) <= atol:
        return xp.real(x)
    return x


def hs_inner(a, b) -> float:
    r"""Hilbert-Schmidt inner product Re[Tr(A^\dagger B)]."""
    xp = _get_array_module(a, b)
    a = _as_array(a, dtype=xp.complex128, xp=xp)
    b = _as_array(b, dtype=xp.complex128, xp=xp)
    _check_same_shape(a, b, "a", "b")
    val = xp.real(xp.trace(dagger(a) @ b))
    return float(_to_scalar(val))


def frobenius_norm(a) -> float:
    """Frobenius norm of an array."""
    xp = _get_array_module(a)
    a = _as_array(a, xp=xp)
    return float(_to_scalar(xp.linalg.norm(a, ord="fro")))


# ============================================================
# Density matrix checks and projections
# ============================================================

def is_hermitian(a, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL) -> bool:
    """Check whether a matrix is Hermitian up to numerical tolerances."""
    xp = _get_array_module(a)
    a = _as_array(a, dtype=xp.complex128, xp=xp)
    _check_square_matrix(a, "a")
    return bool(_to_scalar(xp.allclose(a, dagger(a), atol=atol, rtol=rtol)))


def is_psd(a, atol: float = DEFAULT_ATOL) -> bool:
    """Check whether a Hermitian matrix is positive semidefinite up to tolerance."""
    xp = _get_array_module(a)
    a = hermitian_part(a)
    evals = xp.linalg.eigvalsh(a)
    min_eval = _to_scalar(xp.min(evals))
    return bool(float(min_eval) >= -atol)


def is_density_matrix(
    rho,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    """Check Hermitian, PSD, and unit-trace conditions."""
    xp = _get_array_module(rho)
    rho = _as_array(rho, dtype=xp.complex128, xp=xp)
    _check_square_matrix(rho, "rho")
    if not is_hermitian(rho, atol=atol, rtol=rtol):
        return False
    if not is_psd(rho, atol=atol):
        return False
    tr = _to_scalar(xp.trace(rho))
    return bool(np.isclose(np.real(tr), 1.0, atol=atol, rtol=rtol) and abs(np.imag(tr)) <= atol)


def normalize_trace(a, target_trace: float = 1.0):
    """Rescale a matrix so its trace equals `target_trace`."""
    xp = _get_array_module(a)
    a = _as_array(a, dtype=xp.complex128, xp=xp)
    _check_square_matrix(a, "a")
    tr = _to_scalar(xp.trace(a))
    tr_real = np.real_if_close(tr)
    if abs(float(np.real(tr_real))) < DEFAULT_ATOL:
        raise ValueError("Cannot normalize trace because the matrix trace is numerically zero.")
    return a * (target_trace / tr_real)


def project_vector_to_simplex(v, z: float = 1.0):
    """Euclidean projection onto {x >= 0, sum(x)=z}."""
    xp = _get_array_module(v)
    v = _as_array(v, dtype=float, xp=xp).reshape(-1)
    if z <= 0:
        raise ValueError(f"Simplex radius z must be positive, got {z}.")
    n = int(v.size)
    if n == 0:
        raise ValueError("Input vector must be non-empty.")

    if bool(_to_scalar(xp.all(v >= 0.0))) and bool(
        _to_scalar(xp.isclose(xp.sum(v), z, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL))
    ):
        return v.copy()

    u = xp.sort(v)[::-1]
    cssv = xp.cumsum(u) - z
    ind = xp.arange(1, n + 1)
    cond = u - cssv / ind > 0

    if not bool(_to_scalar(xp.any(cond))):
        return xp.full(n, z / n, dtype=float)

    rho = int(_to_scalar(ind[cond][-1]))
    theta = _to_scalar(cssv[rho - 1]) / rho
    w = xp.maximum(v - theta, 0.0)

    s = float(_to_scalar(xp.sum(w)))
    if s <= 0:
        return xp.full(n, z / n, dtype=float)
    return w * (z / s)


def project_columns_to_simplex(a, z: float = 1.0):
    """Project each column of a matrix onto simplex sum z."""
    xp = _get_array_module(a)
    a = _as_array(a, dtype=float, xp=xp)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got shape {a.shape}.")
    out = xp.empty_like(a, dtype=float)
    for j in range(a.shape[1]):
        out[:, j] = project_vector_to_simplex(a[:, j], z=z)
    return out


def project_rows_to_simplex(a, z: float = 1.0):
    """Project each row of a matrix onto simplex sum z."""
    xp = _get_array_module(a)
    a = _as_array(a, dtype=float, xp=xp)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got shape {a.shape}.")
    out = xp.empty_like(a, dtype=float)
    for i in range(a.shape[0]):
        out[i, :] = project_vector_to_simplex(a[i, :], z=z)
    return out


def project_to_density_matrix(a, target_trace: float = 1.0):
    """Project matrix onto density-matrix set via eigenspectrum clipping + simplex."""
    xp = _get_array_module(a)
    a = hermitian_part(a)
    _check_square_matrix(a, "a")
    if target_trace <= 0:
        raise ValueError(f"target_trace must be positive, got {target_trace}.")

    evals, evecs = xp.linalg.eigh(a)
    evals_proj = project_vector_to_simplex(xp.real(evals), z=target_trace)
    rho = evecs @ xp.diag(evals_proj) @ dagger(evecs)
    return hermitian_part(rho)


def closest_psd(a):
    """Project matrix onto PSD cone while preserving Hermiticity."""
    xp = _get_array_module(a)
    a = hermitian_part(a)
    evals, evecs = xp.linalg.eigh(a)
    evals_clip = xp.maximum(xp.real(evals), 0.0)
    psd = evecs @ xp.diag(evals_clip) @ dagger(evecs)
    return hermitian_part(psd)


def clip_probabilities(
    p,
    floor: float = 0.0,
    renormalize: bool = True,
):
    """Clip probability vector at floor and optionally renormalize."""
    xp = _get_array_module(p)
    p = _as_array(p, dtype=float, xp=xp).reshape(-1)
    if floor < 0:
        raise ValueError(f"floor must be non-negative, got {floor}.")
    q = xp.maximum(p, floor)
    if not renormalize:
        return q
    s = float(_to_scalar(xp.sum(q)))
    if s <= 0:
        return xp.full_like(q, 1.0 / q.size)
    return q / s


def normalize_probability_vector(p, floor: float = 0.0):
    """Normalize a vector to a valid probability vector with optional floor."""
    xp = _get_array_module(p)
    p = _as_array(p, dtype=float, xp=xp).reshape(-1)
    if p.size == 0:
        raise ValueError("Probability vector must be non-empty.")
    q = clip_probabilities(p, floor=floor, renormalize=False)
    s = float(_to_scalar(xp.sum(q)))
    if s <= 0:
        return xp.full(p.size, 1.0 / p.size, dtype=float)
    q = q / s
    if floor > 0:
        q = clip_probabilities(q, floor=floor, renormalize=True)
    return q


def is_column_stochastic(
    c,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    """Check non-negativity and column sums = 1."""
    xp = _get_array_module(c)
    c = _as_array(c, dtype=float, xp=xp)
    if c.ndim != 2:
        return False
    nonneg = bool(_to_scalar(xp.all(c >= -atol)))
    col_sums = xp.sum(c, axis=0)
    sums_ok = bool(_to_scalar(xp.allclose(col_sums, 1.0, atol=atol, rtol=rtol)))
    return nonneg and sums_ok


def project_to_column_stochastic(c):
    """Project matrix onto column-stochastic set (column-wise simplex projection)."""
    return project_columns_to_simplex(c, z=1.0)


# ============================================================
# Tensor / subsystem operations
# ============================================================

def kron_all(operators: Sequence):
    """Kronecker product of a non-empty list of operators."""
    if len(operators) == 0:
        raise ValueError("operators must be non-empty.")
    xp = _get_array_module(operators)
    out = _as_array(operators[0], xp=xp)
    for op in operators[1:]:
        out = xp.kron(out, _as_array(op, xp=xp))
    return out


def permute_subsystems(rho, dims: Sequence[int], perm: Sequence[int]):
    """Permute subsystem order for a density matrix."""
    xp = _get_array_module(rho)
    rho = _as_array(rho, dtype=xp.complex128, xp=xp)
    _check_square_matrix(rho, "rho")

    dims = _validate_dims(dims)
    n = len(dims)
    perm = _validate_indices(perm, n, name="perm")
    if len(perm) != n:
        raise ValueError(f"perm must be a full permutation of 0..{n-1}, got {perm}.")

    dim_total = _prod_int(dims)
    if rho.shape != (dim_total, dim_total):
        raise ValueError(
            f"rho shape {rho.shape} incompatible with dims product {dim_total}."
        )

    tensor = rho.reshape(*dims, *dims)
    axes = list(perm) + [p + n for p in perm]
    tensor_perm = xp.transpose(tensor, axes=axes)
    dims_perm = [dims[p] for p in perm]
    dim_perm = _prod_int(dims_perm)
    return tensor_perm.reshape(dim_perm, dim_perm)


def partial_trace(
    rho,
    dims: Sequence[int],
    keep: Optional[Sequence[int]] = None,
    trace_out: Optional[Sequence[int]] = None,
):
    """
    Partial trace of a composite-system density matrix.

    Exactly one of `keep` or `trace_out` can be omitted. If both are provided,
    they must be complementary.
    """
    xp = _get_array_module(rho)
    rho = _as_array(rho, dtype=xp.complex128, xp=xp)
    _check_square_matrix(rho, "rho")

    dims = _validate_dims(dims)
    n = len(dims)
    total_dim = _prod_int(dims)
    if rho.shape != (total_dim, total_dim):
        raise ValueError(
            f"rho shape {rho.shape} incompatible with dims product {total_dim}."
        )

    if keep is None and trace_out is None:
        raise ValueError("At least one of `keep` or `trace_out` must be provided.")

    all_idx = list(range(n))

    if keep is not None:
        keep_idx = _validate_indices(keep, n, name="keep")
    else:
        keep_idx = [i for i in all_idx if i not in _validate_indices(trace_out, n, "trace_out")]

    if trace_out is not None:
        trace_idx = _validate_indices(trace_out, n, name="trace_out")
    else:
        trace_idx = [i for i in all_idx if i not in keep_idx]

    if set(keep_idx).intersection(trace_idx):
        raise ValueError("keep and trace_out must be disjoint.")
    if set(keep_idx).union(trace_idx) != set(all_idx):
        raise ValueError("keep and trace_out must partition all subsystem indices.")

    keep_dims = [dims[i] for i in keep_idx]
    trace_dims = [dims[i] for i in trace_idx]
    d_keep = _prod_int(keep_dims) if keep_dims else 1
    d_trace = _prod_int(trace_dims) if trace_dims else 1

    tensor = rho.reshape(*dims, *dims)
    axes = keep_idx + trace_idx + [i + n for i in keep_idx] + [i + n for i in trace_idx]
    tensor = xp.transpose(tensor, axes=axes)
    tensor = tensor.reshape(d_keep, d_trace, d_keep, d_trace)

    if d_trace == 1:
        out = tensor[:, 0, :, 0]
    else:
        out = xp.einsum("aibi->ab", tensor)

    return out.reshape(d_keep, d_keep)


def subsystem_dimensions_from_qubits(qubits_per_site: Sequence[int]) -> List[int]:
    """Convert qubits-per-site list into subsystem dimensions (2**q per site)."""
    if len(qubits_per_site) == 0:
        raise ValueError("qubits_per_site must be non-empty.")
    dims: List[int] = []
    for i, q in enumerate(qubits_per_site):
        q_int = int(q)
        if q_int <= 0:
            raise ValueError(
                f"Each entry of qubits_per_site must be positive, got {q_int} at index {i}."
            )
        dims.append(2**q_int)
    return dims


def identity(dim: int):
    """Identity matrix of size dim."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")
    return np.eye(dim, dtype=np.complex128)


def maximally_mixed(dim: int):
    """Maximally mixed state I/dim."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")
    return np.eye(dim, dtype=np.complex128) / dim


# ============================================================
# Self-tests (NumPy-only smoke tests)
# ============================================================

def _self_test_simplex() -> None:
    v = np.array([0.2, -1.0, 3.0, 0.7])
    w = project_vector_to_simplex(v, z=1.0)
    assert np.all(w >= -1e-12)
    assert np.isclose(np.sum(w), 1.0, atol=1e-10)


def _self_test_density_projection() -> None:
    rng = np.random.default_rng(123)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = project_to_density_matrix(a)
    assert is_density_matrix(rho)


def _self_test_partial_trace_bell() -> None:
    psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho = np.outer(psi, np.conjugate(psi))
    red0 = partial_trace(rho, dims=[2, 2], keep=[0])
    red1 = partial_trace(rho, dims=[2, 2], keep=[1])
    mm = np.eye(2) / 2.0
    assert np.allclose(red0, mm, atol=1e-10)
    assert np.allclose(red1, mm, atol=1e-10)


def _self_test_permutation() -> None:
    rho = np.eye(6, dtype=np.complex128) / 6.0
    rho_perm = permute_subsystems(rho, dims=[2, 3], perm=[1, 0])
    assert rho_perm.shape == (6, 6)
    assert np.isclose(np.trace(rho_perm), 1.0)


def _self_test_column_stochastic() -> None:
    c = np.array([[0.7, 0.2], [0.3, 0.8]])
    assert is_column_stochastic(c)
    bad = np.array([[1.1, -0.1], [-0.1, 1.1]])
    proj = project_to_column_stochastic(bad)
    assert is_column_stochastic(proj)


def run_self_tests(verbose: bool = True) -> None:
    """Run internal self-tests."""
    tests = [
        _self_test_simplex,
        _self_test_density_projection,
        _self_test_partial_trace_bell,
        _self_test_permutation,
        _self_test_column_stochastic,
    ]
    for fn in tests:
        fn()
        if verbose:
            print(f"[core_ops] PASSED: {fn.__name__}")
    if verbose:
        print("[core_ops] All self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
