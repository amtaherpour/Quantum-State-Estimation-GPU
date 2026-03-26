from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch


# ============================================================
# Numerical defaults
# ============================================================

DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-8
DEFAULT_PROB_FLOOR = 1e-12

DEFAULT_REAL_DTYPE = torch.float64
DEFAULT_COMPLEX_DTYPE = torch.complex128


# ============================================================
# Internal dtype / device helpers
# ============================================================

def _infer_device(*xs: Any) -> Optional[torch.device]:
    """Infer a torch device from the first tensor argument, if present."""
    for x in xs:
        if isinstance(x, torch.Tensor):
            return x.device
    return None


def _infer_real_dtype(x: Any) -> torch.dtype:
    """Infer a real dtype while preserving tensor precision when possible."""
    if isinstance(x, torch.Tensor):
        if x.is_complex():
            return torch.float32 if x.dtype == torch.complex64 else torch.float64
        if x.dtype.is_floating_point:
            return x.dtype
    return DEFAULT_REAL_DTYPE


def _infer_complex_dtype(x: Any) -> torch.dtype:
    """Infer a complex dtype while preserving tensor precision when possible."""
    if isinstance(x, torch.Tensor):
        if x.is_complex():
            return x.dtype
        if x.dtype == torch.float32:
            return torch.complex64
        if x.dtype == torch.float64:
            return torch.complex128
    return DEFAULT_COMPLEX_DTYPE


# ============================================================
# Basic validation helpers
# ============================================================

def _as_torch_tensor(
    x: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    copy: bool = False,
) -> torch.Tensor:
    """
    Convert input to a torch tensor.

    Notes
    -----
    - If `x` is already a tensor, its device is preserved unless `device` is
      explicitly provided.
    - If `dtype` is None, the existing dtype is preserved for tensors.
    """
    if isinstance(x, torch.Tensor):
        out = x.to(
            dtype=x.dtype if dtype is None else dtype,
            device=x.device if device is None else device,
        )
        return out.clone() if copy else out

    out = torch.as_tensor(x, dtype=dtype, device=device)
    return out.clone() if copy else out


def _check_square_matrix(a: torch.Tensor, name: str = "matrix") -> None:
    """Raise ValueError if `a` is not a square 2D tensor."""
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor, got shape {tuple(a.shape)}.")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be square, got shape {tuple(a.shape)}.")


def _check_same_shape(
    a: torch.Tensor,
    b: torch.Tensor,
    name_a: str = "a",
    name_b: str = "b",
) -> None:
    """Raise ValueError if two tensors do not have the same shape."""
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape, got "
            f"{tuple(a.shape)} and {tuple(b.shape)}."
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

def dagger(a: Any) -> torch.Tensor:
    """
    Conjugate transpose of a matrix.

    Parameters
    ----------
    a :
        Input matrix-like object.

    Returns
    -------
    torch.Tensor
        Conjugate transpose of `a`.
    """
    a = _as_torch_tensor(a, device=_infer_device(a))
    if a.ndim < 2:
        raise ValueError(f"a must have at least 2 dimensions, got shape {tuple(a.shape)}.")
    return a.transpose(-2, -1).conj()


def hermitian_part(a: Any) -> torch.Tensor:
    r"""
    Return the Hermitian part of a square matrix:
        (A + A^\dagger) / 2.
    """
    a = _as_torch_tensor(a, dtype=_infer_complex_dtype(a), device=_infer_device(a))
    _check_square_matrix(a, "a")
    return 0.5 * (a + dagger(a))


def antihermitian_part(a: Any) -> torch.Tensor:
    r"""
    Return the anti-Hermitian part of a square matrix:
        (A - A^\dagger) / 2.
    """
    a = _as_torch_tensor(a, dtype=_infer_complex_dtype(a), device=_infer_device(a))
    _check_square_matrix(a, "a")
    return 0.5 * (a - dagger(a))


def real_if_close_scalar(x: Any, atol: float = DEFAULT_ATOL) -> float | complex:
    """
    Return the real part if the imaginary part is numerically negligible.
    """
    t = _as_torch_tensor(x, device=_infer_device(x))
    if t.numel() != 1:
        raise ValueError(f"x must contain exactly one scalar value, got shape {tuple(t.shape)}.")
    value = t.reshape(()).item()
    if isinstance(value, complex) and abs(value.imag) <= atol:
        return float(value.real)
    return value


def real_if_close_array(x: Any, atol: float = DEFAULT_ATOL) -> torch.Tensor:
    """
    Return a real tensor if all imaginary parts are numerically negligible.
    Otherwise return the original complex-valued tensor.
    """
    t = _as_torch_tensor(x, device=_infer_device(x))
    if t.is_complex():
        max_imag = (
            torch.max(torch.abs(t.imag))
            if t.numel() > 0
            else torch.tensor(0.0, dtype=torch.float64, device=t.device)
        )
        if float(max_imag.item()) <= atol:
            return t.real
    return t


def hs_inner(a: Any, b: Any) -> float:
    r"""
    Hilbert-Schmidt inner product Re[Tr(A^\dagger B)].

    This is the natural real inner product for complex matrices when using
    Frobenius geometry.

    Returns
    -------
    float
    """
    device = _infer_device(a, b)
    template = a if isinstance(a, torch.Tensor) else b
    dtype = _infer_complex_dtype(template)
    a_t = _as_torch_tensor(a, dtype=dtype, device=device)
    b_t = _as_torch_tensor(b, dtype=dtype, device=device)
    _check_same_shape(a_t, b_t, "a", "b")
    return float(torch.real(torch.trace(dagger(a_t) @ b_t)).item())


def frobenius_norm(a: Any) -> float:
    """
    Frobenius norm of an array / tensor.
    """
    a_t = _as_torch_tensor(a, device=_infer_device(a))
    return float(torch.linalg.vector_norm(a_t.reshape(-1)).item())


# ============================================================
# Density matrix checks and projections
# ============================================================

def is_hermitian(a: Any, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL) -> bool:
    """
    Check whether a matrix is Hermitian up to numerical tolerances.
    """
    a_t = _as_torch_tensor(a, dtype=_infer_complex_dtype(a), device=_infer_device(a))
    _check_square_matrix(a_t, "a")
    return bool(torch.allclose(a_t, dagger(a_t), atol=atol, rtol=rtol))


def is_psd(a: Any, atol: float = DEFAULT_ATOL) -> bool:
    """
    Check whether a Hermitian matrix is positive semidefinite up to tolerance.

    A small negative eigenvalue above numerical noise is allowed.
    """
    a_t = hermitian_part(a)
    evals = torch.linalg.eigvalsh(a_t)
    return bool(torch.min(evals).item() >= -atol)


def is_density_matrix(
    rho: Any,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    """
    Check whether a matrix is a valid density matrix:
    Hermitian, PSD, and unit trace.
    """
    rho_t = _as_torch_tensor(rho, dtype=_infer_complex_dtype(rho), device=_infer_device(rho))
    _check_square_matrix(rho_t, "rho")
    if not is_hermitian(rho_t, atol=atol, rtol=rtol):
        return False
    if not is_psd(rho_t, atol=atol):
        return False
    tr = torch.trace(rho_t)
    return bool(
        abs(torch.imag(tr).item()) <= atol
        and torch.isclose(
            torch.real(tr),
            torch.tensor(1.0, dtype=torch.real(rho_t).dtype, device=rho_t.device),
            atol=atol,
            rtol=rtol,
        ).item()
    )


def normalize_trace(a: Any, target_trace: float = 1.0) -> torch.Tensor:
    """
    Rescale a matrix so its trace equals `target_trace`.

    This does NOT enforce Hermiticity or PSD. It is only a trace rescaling.
    """
    a_t = _as_torch_tensor(a, dtype=_infer_complex_dtype(a), device=_infer_device(a))
    _check_square_matrix(a_t, "a")
    tr = torch.trace(a_t)
    if abs(tr.item()) < DEFAULT_ATOL:
        raise ValueError("Cannot normalize trace because the matrix trace is numerically zero.")
    return a_t * (target_trace / tr)


def _project_last_dim_to_simplex(x: Any, z: float = 1.0) -> torch.Tensor:
    """
    Vectorized Euclidean projection onto the simplex along the last dimension.
    """
    if z <= 0:
        raise ValueError(f"Simplex radius z must be positive, got {z}.")
    x_t = _as_torch_tensor(x, dtype=_infer_real_dtype(x), device=_infer_device(x))
    if x_t.ndim == 0:
        raise ValueError("Input must have at least one dimension.")
    n = x_t.shape[-1]
    if n == 0:
        raise ValueError("Last dimension must be non-empty.")

    u, _ = torch.sort(x_t, dim=-1, descending=True)
    cssv = torch.cumsum(u, dim=-1) - z
    ind = torch.arange(1, n + 1, device=x_t.device, dtype=x_t.dtype)
    cond = u - cssv / ind > 0

    rho = cond.to(torch.int64).sum(dim=-1).clamp(min=1)
    theta = cssv.gather(-1, (rho - 1).unsqueeze(-1)) / rho.unsqueeze(-1).to(x_t.dtype)

    w = torch.clamp(x_t - theta, min=0.0)
    s = w.sum(dim=-1, keepdim=True)

    fallback = torch.full_like(w, z / n)
    scale = z / s.clamp_min(torch.finfo(w.dtype).tiny)
    return torch.where(s > 0.0, w * scale, fallback)


def project_vector_to_simplex(v: Any, z: float = 1.0) -> torch.Tensor:
    """
    Euclidean projection of a real vector onto the probability simplex:
        {x >= 0, sum(x) = z}

    Uses the standard sorting-based algorithm.

    Parameters
    ----------
    v :
        Real input vector.
    z :
        Desired simplex sum. Must be positive.

    Returns
    -------
    torch.Tensor
        Projected vector.
    """
    v_t = _as_torch_tensor(v, dtype=_infer_real_dtype(v), device=_infer_device(v)).reshape(-1)
    if z <= 0:
        raise ValueError(f"Simplex radius z must be positive, got {z}.")
    if v_t.numel() == 0:
        raise ValueError("Input vector must be non-empty.")

    z_t = torch.tensor(float(z), dtype=v_t.dtype, device=v_t.device)
    if torch.all(v_t >= 0.0) and torch.isclose(v_t.sum(), z_t, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL):
        return v_t.clone()

    return _project_last_dim_to_simplex(v_t, z=z)


def project_columns_to_simplex(a: Any, z: float = 1.0) -> torch.Tensor:
    """
    Project each column of a real matrix onto the simplex
        {x >= 0, sum(x) = z}.
    """
    a_t = _as_torch_tensor(a, dtype=_infer_real_dtype(a), device=_infer_device(a))
    if a_t.ndim != 2:
        raise ValueError(f"a must be 2D, got shape {tuple(a_t.shape)}.")
    return _project_last_dim_to_simplex(a_t.transpose(0, 1), z=z).transpose(0, 1)


def project_rows_to_simplex(a: Any, z: float = 1.0) -> torch.Tensor:
    """
    Project each row of a real matrix onto the simplex
        {x >= 0, sum(x) = z}.
    """
    a_t = _as_torch_tensor(a, dtype=_infer_real_dtype(a), device=_infer_device(a))
    if a_t.ndim != 2:
        raise ValueError(f"a must be 2D, got shape {tuple(a_t.shape)}.")
    return _project_last_dim_to_simplex(a_t, z=z)


def project_to_density_matrix(a: Any, target_trace: float = 1.0) -> torch.Tensor:
    """
    Project a square complex matrix onto the set of density matrices
    (Hermitian PSD matrices with specified trace) in Frobenius geometry.

    Procedure:
    1) Hermitian symmetrization
    2) Eigenvalue decomposition
    3) Project eigenvalues onto simplex with sum = target_trace
    4) Reconstruct matrix

    Parameters
    ----------
    a :
        Square complex matrix.
    target_trace :
        Desired trace, typically 1.0.

    Returns
    -------
    torch.Tensor
        Projected density matrix.
    """
    a_t = _as_torch_tensor(a, dtype=_infer_complex_dtype(a), device=_infer_device(a))
    _check_square_matrix(a_t, "a")
    if target_trace <= 0:
        raise ValueError(f"target_trace must be positive, got {target_trace}.")

    h = hermitian_part(a_t)
    evals, evecs = torch.linalg.eigh(h)
    evals_proj = project_vector_to_simplex(evals.real, z=target_trace)
    rho = (evecs * evals_proj.unsqueeze(0)) @ dagger(evecs)
    rho = hermitian_part(rho)
    rho = normalize_trace(rho, target_trace=target_trace)

    evals2, evecs2 = torch.linalg.eigh(rho)
    evals2 = torch.clamp(evals2.real, min=0.0)
    s = evals2.sum()
    if float(s.item()) <= 0.0:
        raise ValueError("Projected eigenvalues have non-positive total mass.")
    evals2 = evals2 * (target_trace / s)
    rho = (evecs2 * evals2.unsqueeze(0)) @ dagger(evecs2)
    rho = hermitian_part(rho)
    return rho


def closest_psd(a: Any) -> torch.Tensor:
    """
    Project a square matrix onto the PSD cone by Hermitian symmetrization
    followed by clipping negative eigenvalues to zero.

    This does NOT enforce unit trace.
    """
    a_t = _as_torch_tensor(a, dtype=_infer_complex_dtype(a), device=_infer_device(a))
    _check_square_matrix(a_t, "a")
    h = hermitian_part(a_t)
    evals, evecs = torch.linalg.eigh(h)
    evals = torch.clamp(evals.real, min=0.0)
    return hermitian_part((evecs * evals.unsqueeze(0)) @ dagger(evecs))


# ============================================================
# Probability helpers
# ============================================================

def clip_probabilities(
    p: Any,
    floor: float = DEFAULT_PROB_FLOOR,
    renormalize: bool = True,
) -> torch.Tensor:
    """
    Clip a probability vector away from zero and optionally renormalize.

    Useful before log-likelihood evaluations.
    """
    p_t = _as_torch_tensor(p, dtype=_infer_real_dtype(p), device=_infer_device(p)).reshape(-1)
    if floor <= 0:
        raise ValueError(f"floor must be positive, got {floor}.")
    q = torch.clamp(p_t, min=float(floor))
    if renormalize:
        s = q.sum()
        if float(s.item()) <= 0.0:
            raise ValueError("Cannot renormalize probabilities with non-positive total mass.")
        q = q / s
    return q


def normalize_probability_vector(p: Any, floor: float = 0.0) -> torch.Tensor:
    """
    Project a real vector onto the simplex after optional lower clipping.
    """
    p_t = _as_torch_tensor(p, dtype=_infer_real_dtype(p), device=_infer_device(p)).reshape(-1)
    if floor < 0:
        raise ValueError(f"floor must be non-negative, got {floor}.")
    q = torch.clamp(p_t, min=float(floor))
    return project_vector_to_simplex(q, z=1.0)


# ============================================================
# Stochastic-matrix helpers
# ============================================================

def is_column_stochastic(
    c: Any,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    """
    Check whether a matrix is column-stochastic:
    nonnegative and each column sums to 1.
    """
    c_t = _as_torch_tensor(c, dtype=_infer_real_dtype(c), device=_infer_device(c))
    if c_t.ndim != 2:
        return False
    if torch.any(c_t < -atol):
        return False
    col_sums = torch.sum(c_t, dim=0)
    return bool(
        torch.allclose(
            col_sums,
            torch.ones_like(col_sums),
            atol=atol,
            rtol=rtol,
        )
    )


def project_to_column_stochastic(c: Any) -> torch.Tensor:
    """
    Project a real matrix onto the set of column-stochastic matrices
    by projecting each column onto the simplex.
    """
    c_t = _as_torch_tensor(c, dtype=_infer_real_dtype(c), device=_infer_device(c))
    if c_t.ndim != 2:
        raise ValueError(f"c must be 2D, got shape {tuple(c_t.shape)}.")
    return project_columns_to_simplex(c_t, z=1.0)


# ============================================================
# Tensor / subsystem operations
# ============================================================

def kron_all(operators: Sequence[Any]) -> torch.Tensor:
    """
    Kronecker product of a sequence of arrays.

    Parameters
    ----------
    operators :
        Non-empty sequence of matrices or vectors.

    Returns
    -------
    torch.Tensor
        Kronecker product in left-to-right order.
    """
    if len(operators) == 0:
        raise ValueError("operators must be a non-empty sequence.")

    device = _infer_device(*operators)
    template = next((op for op in operators if isinstance(op, torch.Tensor)), operators[0])
    dtype = _infer_complex_dtype(template)

    out = _as_torch_tensor(operators[0], dtype=dtype, device=device)
    for op in operators[1:]:
        out = torch.kron(out, _as_torch_tensor(op, dtype=dtype, device=device))
    return out


def permute_subsystems(rho: Any, dims: Sequence[int], perm: Sequence[int]) -> torch.Tensor:
    """
    Permute subsystem order of a density matrix.

    If rho acts on H_1 ⊗ ... ⊗ H_n with subsystem dimensions `dims`,
    then `perm` gives the new subsystem order.

    Example
    -------
    dims = [2, 3, 2], perm = [2, 0, 1]
    means:
        H_1 ⊗ H_2 ⊗ H_3  ->  H_3 ⊗ H_1 ⊗ H_2

    Parameters
    ----------
    rho :
        Square matrix of shape (prod(dims), prod(dims)).
    dims :
        Subsystem dimensions.
    perm :
        A permutation of range(len(dims)).

    Returns
    -------
    torch.Tensor
        Permuted density matrix.
    """
    rho_t = _as_torch_tensor(rho, dtype=_infer_complex_dtype(rho), device=_infer_device(rho))
    _check_square_matrix(rho_t, "rho")
    dims_list = _validate_dims(dims, "dims")
    n = len(dims_list)
    perm_list = _validate_indices(perm, n, "perm")
    if sorted(perm_list) != list(range(n)):
        raise ValueError(f"perm must be a permutation of 0..{n - 1}. Got {perm_list}.")

    d_total = _prod_int(dims_list)
    if tuple(rho_t.shape) != (d_total, d_total):
        raise ValueError(
            f"rho shape {tuple(rho_t.shape)} is incompatible with subsystem dimensions "
            f"{dims_list} (total dimension {d_total})."
        )

    if n == 1:
        return rho_t.clone()

    tensor = rho_t.reshape(*dims_list, *dims_list)
    axes = list(perm_list) + [p + n for p in perm_list]
    permuted = tensor.permute(*axes)
    new_dims = [dims_list[p] for p in perm_list]
    d_new = _prod_int(new_dims)
    return permuted.reshape(d_new, d_new)


def partial_trace(
    rho: Any,
    dims: Sequence[int],
    keep: Optional[Sequence[int]] = None,
    trace_out: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Partial trace over selected subsystems.

    Exactly one of `keep` or `trace_out` must be provided.

    Parameters
    ----------
    rho :
        Square density matrix or operator on the tensor-product space with
        subsystem dimensions `dims`.
    dims :
        Subsystem dimensions.
    keep :
        Indices of subsystems to keep, in the desired output order.
    trace_out :
        Indices of subsystems to trace out.

    Returns
    -------
    torch.Tensor
        Reduced operator on the kept subsystems.

    Notes
    -----
    - If `keep=[]`, the result is a 1x1 matrix containing Tr(rho).
    - The output subsystem order follows the order specified in `keep`.
    """
    rho_t = _as_torch_tensor(rho, dtype=_infer_complex_dtype(rho), device=_infer_device(rho))
    _check_square_matrix(rho_t, "rho")
    dims_list = _validate_dims(dims, "dims")
    n = len(dims_list)

    if (keep is None) == (trace_out is None):
        raise ValueError("Exactly one of `keep` or `trace_out` must be provided.")

    d_total = _prod_int(dims_list)
    if tuple(rho_t.shape) != (d_total, d_total):
        raise ValueError(
            f"rho shape {tuple(rho_t.shape)} is incompatible with dims {dims_list} "
            f"(total dimension {d_total})."
        )

    if keep is not None:
        keep_list = _validate_indices(keep, n, "keep")
        trace_out_list = [i for i in range(n) if i not in keep_list]
        keep_desired_order = list(keep_list)
    else:
        trace_out_list = _validate_indices(trace_out, n, "trace_out")
        keep_desired_order = [i for i in range(n) if i not in trace_out_list]

    if len(keep_desired_order) == 0:
        return torch.tensor([[torch.trace(rho_t)]], dtype=rho_t.dtype, device=rho_t.device)

    keep_canonical_order = [i for i in range(n) if i not in trace_out_list]
    current_dims = dims_list.copy()
    tensor = rho_t.reshape(*dims_list, *dims_list)

    for ax in sorted(trace_out_list, reverse=True):
        n_cur = len(current_dims)
        tensor = torch.diagonal(tensor, offset=0, dim1=ax, dim2=ax + n_cur).sum(dim=-1)
        current_dims.pop(ax)

    reduced = tensor.reshape(_prod_int(current_dims), _prod_int(current_dims))

    if keep_desired_order != keep_canonical_order and len(current_dims) > 1:
        perm_local = [keep_canonical_order.index(i) for i in keep_desired_order]
        reduced = permute_subsystems(reduced, current_dims, perm_local)

    return reduced


def subsystem_dimensions_from_qubits(qubits_per_site: Sequence[int]) -> List[int]:
    """
    Convert a list of qubit counts per site to subsystem Hilbert-space dimensions.

    Example
    -------
    [1, 2, 1] -> [2, 4, 2]
    """
    q = [int(v) for v in qubits_per_site]
    if any(v < 0 for v in q):
        raise ValueError(f"qubits_per_site must be non-negative, got {q}.")
    return [2 ** v for v in q]


# ============================================================
# Small utility constructors
# ============================================================

def identity(
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Complex identity matrix of size `dim`."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")
    return torch.eye(dim, dtype=DEFAULT_COMPLEX_DTYPE if dtype is None else dtype, device=device)


def maximally_mixed(
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Maximally mixed state I / dim."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")
    return identity(dim, dtype=dtype, device=device) / dim


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_simplex() -> None:
    v = torch.tensor([0.2, -0.3, 2.0, 0.1], dtype=torch.float64)
    w = project_vector_to_simplex(v, z=1.0)
    assert bool(torch.all(w >= -1e-12))
    assert bool(torch.isclose(torch.sum(w), torch.tensor(1.0, dtype=w.dtype), atol=1e-10))


def _self_test_density_projection() -> None:
    a = torch.tensor([[0.7, 2.0 + 1.0j], [2.0 - 1.0j, -0.2]], dtype=torch.complex128)
    rho = project_to_density_matrix(a)
    assert is_density_matrix(rho)
    assert bool(torch.isclose(torch.real(torch.trace(rho)), torch.tensor(1.0, dtype=torch.float64), atol=1e-10))


def _self_test_partial_trace_bell() -> None:
    ket = torch.tensor([1, 0, 0, 1], dtype=torch.complex128) / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
    rho = torch.outer(ket, ket.conj())
    red0 = partial_trace(rho, dims=[2, 2], keep=[0])
    red1 = partial_trace(rho, dims=[2, 2], keep=[1])
    target = torch.eye(2, dtype=torch.complex128) / 2.0
    assert bool(torch.allclose(red0, target, atol=1e-10))
    assert bool(torch.allclose(red1, target, atol=1e-10))


def _self_test_permutation() -> None:
    dims = [2, 3]
    rho = torch.arange(36, dtype=torch.float64).reshape(6, 6).to(torch.complex128)
    rho_perm = permute_subsystems(rho, dims=dims, perm=[1, 0])
    rho_back = permute_subsystems(rho_perm, dims=[3, 2], perm=[1, 0])
    assert bool(torch.allclose(rho, rho_back))


def _self_test_column_stochastic() -> None:
    c = torch.tensor([[1.2, -0.1], [-0.2, 1.1]], dtype=torch.float64)
    cp = project_to_column_stochastic(c)
    assert is_column_stochastic(cp)


def _self_test_gpu_smoke() -> None:
    if not torch.cuda.is_available():
        return
    a = torch.tensor([[0.7, 2.0 + 1.0j], [2.0 - 1.0j, -0.2]], dtype=torch.complex128, device="cuda")
    rho = project_to_density_matrix(a)
    assert rho.device.type == "cuda"
    assert is_density_matrix(rho)


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a small suite of smoke tests for this module.
    """
    tests = [
        ("simplex projection", _self_test_simplex),
        ("density projection", _self_test_density_projection),
        ("partial trace (Bell state)", _self_test_partial_trace_bell),
        ("subsystem permutation", _self_test_permutation),
        ("column stochastic projection", _self_test_column_stochastic),
        ("gpu smoke", _self_test_gpu_smoke),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All core_ops self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
