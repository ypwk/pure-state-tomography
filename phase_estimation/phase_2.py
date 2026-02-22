from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import src.qutils as qutils
from src.noise_modeling import make_custom_noise_model


def reconstruct_rho_svt(
    A_stack: np.ndarray,
    y: np.ndarray,
    *,
    tau: float = 5.0,
    delta: float = 1e-6,
    max_iters: int = 500,
) -> np.ndarray:
    """
    Reconstruct rho using Singular Value Thresholding (SVT),
    as in Gross et al., arXiv:0909.3304.

    Solves:
        min tau * ||rho||_* + 0.5 * ||rho||_F^2
        s.t. |Tr(A_j rho) - y_j| <= delta

    Returns:
        PSD, trace-1 density matrix.
    """
    M, K, _ = A_stack.shape
    rho = np.zeros((K, K), dtype=complex)
    Y = np.zeros_like(rho)

    A_stack_h = A_stack.conj().transpose(0, 2, 1)

    for _ in range(max_iters):
        # Measurement residual
        pred = np.real(np.einsum("mab,ba->m", A_stack, rho))
        residual = pred - y

        # Dual update
        grad = np.einsum("m,mab->ab", residual, A_stack_h)
        Y -= grad

        # Singular value thresholding
        U, s, Vh = np.linalg.svd(Y, full_matrices=False)
        s_thresh = np.maximum(s - tau, 0.0)
        rho = (U * s_thresh) @ Vh

        # Enforce Hermiticity
        rho = 0.5 * (rho + rho.conj().T)

        # Trace normalization
        tr = np.trace(rho).real
        if tr > 0:
            rho /= tr

        # PSD projection (numerical safety)
        evals, evecs = np.linalg.eigh(rho)
        evals = np.clip(evals, 0.0, None)
        if evals.sum() > 0:
            rho = (evecs * evals) @ evecs.conj().T
            rho /= np.trace(rho).real

        if np.linalg.norm(residual) <= delta:
            break

    return rho

def recover_coefficients_pauli_cs(
    state_prep_circuit,
    *,
    support_indices: Sequence[int],
    num_measurements: int,
    shots: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Recover a K-sparse state using Pauli-expectation compressed sensing on known support.

    Args:
        state_prep_circuit: Qiskit circuit preparing the unknown n-qubit state.
        support_indices: Computational-basis support (length K).
        num_measurements: Number of random Pauli settings to sample.
        shots: Shots per Pauli setting.
        backend: Optional simulator/backend. If None, uses AerSimulator with the
            default custom noise model from run_general_experiments.py.
        seed: RNG seed for reproducibility.
        solver: Reconstruction solver name. Currently supports "iht" and "pgd".

    Returns:
        Full statevector estimate of shape (2^n,), with nonzeros restricted to
        support_indices and unit norm.
    """
    num_qubits = int(state_prep_circuit.num_qubits)
    support = _validate_support_indices(support_indices, num_qubits)
    if num_measurements <= 0:
        raise ValueError("num_measurements must be positive")
    if shots <= 0:
        raise ValueError("shots must be positive")

    rng = np.random.default_rng(seed)
    paulis = sample_random_paulis(
        num_qubits=num_qubits,
        num_measurements=int(num_measurements),
        rng=rng,
    )
    y = measure_pauli_expectations(
        state_prep_circuit,
        pauli_strings=paulis,
        shots=int(shots),
        seed=seed,
    )

    A_stack = np.stack(
        [
            restrict_pauli_to_support(
                p,
                support_indices=support,
                num_qubits=num_qubits,
            )
            for p in paulis
        ],
        axis=0,
    )

    rho = reconstruct_rho_svt(
        A_stack,
        y,
        tau=5.0,
        delta=1e-6,
        max_iters=500,
    )

    x_support = principal_eigenvector(rho)
    x_support = fix_global_phase(x_support)

    full_dim = 1 << num_qubits
    full_state = np.zeros(full_dim, dtype=complex)
    for amp, idx in zip(x_support, support):
        full_state[idx] = amp

    norm = float(np.linalg.norm(full_state))
    if norm <= 0.0:
        raise RuntimeError("Recovered state has zero norm")
    full_state /= norm
    return full_state

def unique_gate_names(circuit: QuantumCircuit) -> set[str]:
    return {instruction.name for instruction, _, _ in circuit.data}

def measure_pauli_expectations(
    state_prep_circuit,
    *,
    pauli_strings: Sequence[str],
    shots: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Estimate y[j] ~= <Psi|P_j|Psi> from sampled counts after basis rotations.

    Pauli-string convention: pauli[q] applies to qubit q (q=0 is least-significant
    computational bit index in basis state integer encoding).
    """
    if shots <= 0:
        raise ValueError("shots must be positive")

    num_qubits = int(state_prep_circuit.num_qubits)
    if not pauli_strings:
        return np.zeros(0, dtype=float)

    noise_model = make_custom_noise_model(
        p_1q=4.239e-4,
        p_2q=3.416e-3,
        p_meas=1e-2,
        coherent_phase=0.0,
        n_qubits=state_prep_circuit.num_qubits
    )
    aer_sim = AerSimulator(
        # noise_model=noise_model,
        method="statevector",
        # device="CPU",
    )

    expectations = np.zeros(len(pauli_strings), dtype=float)
    print(len(pauli_strings))
    for j, pauli in enumerate(pauli_strings):
        _validate_pauli_string(pauli, num_qubits)
        meas_circuit = _build_pauli_measurement_circuit(state_prep_circuit, pauli)
        probs = qutils.run_circuit(
            aer_sim,
            meas_circuit,
            shots=int(shots),
            batch_size=1,
            base_seed=None if seed is None else int(seed) + j,
        )[0]
        expectations[j] = _expectation_from_probabilities(pauli, probs)

    return expectations


def restrict_pauli_to_support(
    pauli: str,
    *,
    support_indices: Sequence[int],
    num_qubits: int,
) -> np.ndarray:
    """
    Build A = S^dagger P S on support, where A[a,b] = <s_a|P|s_b>.

    Pauli-string convention: pauli[q] applies to qubit q.
    """
    support = _validate_support_indices(support_indices, int(num_qubits))
    _validate_pauli_string(pauli, int(num_qubits))

    k = len(support)
    A = np.zeros((k, k), dtype=complex)
    for a, s_a in enumerate(support):
        for b, s_b in enumerate(support):
            A[a, b] = _pauli_element_on_basis_pair(pauli, s_a, s_b, int(num_qubits))

    A = project_to_hermitian(A)
    return A


def sample_random_paulis(
    num_qubits: int,
    num_measurements: int,
    rng,
) -> list[str]:
    """
    Sample random n-qubit Pauli strings.

    Distribution per qubit: P(I)=0.10, P(X)=P(Y)=P(Z)=0.30.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if num_measurements <= 0:
        raise ValueError("num_measurements must be positive")

    alphabet = np.array(["I", "X", "Y", "Z"])
    probs = np.array([0.1, 0.3, 0.3, 0.3], dtype=float)

    out: list[str] = []
    for _ in range(int(num_measurements)):
        chars = rng.choice(alphabet, size=int(num_qubits), p=probs)
        out.append("".join(chars.tolist()))
    return out


def reconstruct_rho_projected_gradient(
    A_stack: np.ndarray,
    y: np.ndarray,
    *,
    solver: Literal["iht", "pgd"] = "iht",
    max_iters: int = 300,
    step_size: float | None = None,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Reconstruct rho via projected gradient on linear constraints y_j ~= Tr(A_j rho).

    For "iht", each iteration projects onto Hermitian -> PSD trace-1 -> rank-1.
    For "pgd", rank-1 projection is skipped during iterations and only done at end.
    """
    if A_stack.ndim != 3:
        raise ValueError("A_stack must have shape (M, K, K)")
    M, K, K2 = A_stack.shape
    if K != K2:
        raise ValueError("A_stack must contain square matrices")
    y = np.asarray(y, dtype=float)
    if y.shape != (M,):
        raise ValueError("y must have shape (M,)")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")

    rho = np.eye(K, dtype=complex) / float(K)

    if step_size is None:
        frob_sq = float(np.sum(np.linalg.norm(A_stack, axis=(1, 2)) ** 2))
        lipschitz = max(frob_sq, 1.0)
        eta = 1.0 / lipschitz
    else:
        eta = float(step_size)
        if eta <= 0.0:
            raise ValueError("step_size must be positive")

    prev_err = np.inf
    apply_rank1_each_iter = solver.lower() == "iht"
    if solver.lower() not in {"iht", "pgd"}:
        raise ValueError("solver must be 'iht' or 'pgd'")

    for _ in range(int(max_iters)):
        pred = np.real(np.einsum("mab,ba->m", A_stack, rho, optimize=True))
        residual = pred - y
        grad = np.einsum("m,mab->ab", residual, A_stack, optimize=True)

        rho = rho - eta * grad
        rho = project_to_hermitian(rho)
        rho = project_to_psd_trace1(rho)
        if apply_rank1_each_iter:
            rho = project_to_rank1(rho)

        err = float(np.linalg.norm(residual))
        if abs(prev_err - err) <= tol:
            break
        prev_err = err

    if not apply_rank1_each_iter:
        rho = project_to_rank1(project_to_psd_trace1(project_to_hermitian(rho)))

    return rho


def project_to_hermitian(rho: np.ndarray) -> np.ndarray:
    """Project a matrix to Hermitian by averaging with its conjugate transpose."""
    return 0.5 * (rho + rho.conj().T)


def project_to_psd_trace1(rho: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Project Hermitian matrix to PSD cone and normalize to trace one."""
    rho_h = project_to_hermitian(rho)
    evals, evecs = np.linalg.eigh(rho_h)
    evals = np.clip(np.real(evals), 0.0, None)
    tr = float(np.sum(evals))
    if tr <= eps:
        d = rho.shape[0]
        return np.eye(d, dtype=complex) / float(d)
    evals /= tr
    return (evecs * evals) @ evecs.conj().T


def project_to_rank1(rho: np.ndarray) -> np.ndarray:
    """Keep principal eigenpair and return |v><v| (trace one, PSD, rank one)."""
    rho_h = project_to_hermitian(rho)
    evals, evecs = np.linalg.eigh(rho_h)
    idx = int(np.argmax(np.real(evals)))
    vec = evecs[:, idx]
    nrm = float(np.linalg.norm(vec))
    if nrm <= 0.0:
        d = rho.shape[0]
        return np.eye(d, dtype=complex) / float(d)
    vec = vec / nrm
    return np.outer(vec, vec.conj())


def principal_eigenvector(rho: np.ndarray) -> np.ndarray:
    """Return normalized eigenvector associated with largest eigenvalue."""
    rho_h = project_to_hermitian(np.asarray(rho, dtype=complex))
    evals, evecs = np.linalg.eigh(rho_h)
    idx = int(np.argmax(np.real(evals)))
    vec = np.asarray(evecs[:, idx], dtype=complex)
    nrm = float(np.linalg.norm(vec))
    if nrm <= 0.0:
        raise RuntimeError("Principal eigenvector has zero norm")
    return vec / nrm


def fix_global_phase(vec: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Fix global phase by making the first nonzero component real and nonnegative."""
    out = np.asarray(vec, dtype=complex).copy()
    idx = None
    for i, val in enumerate(out):
        if abs(val) > eps:
            idx = i
            break
    if idx is None:
        return out
    phase = np.angle(out[idx])
    out *= np.exp(-1j * phase)
    if np.real(out[idx]) < 0:
        out *= -1.0
    return out


def phase2_recover_coefficients(
    state_prep_circuit,
    *,
    support_indices: Sequence[int],
    shots: int,
    backend=None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Backward-compatible wrapper returning amplitudes on support only.

    Uses a default compressed-sensing measurement count of max(8, 4K*ceil(log2(K+1))).
    """
    support = _validate_support_indices(support_indices, state_prep_circuit.num_qubits)
    K = len(support)
    default_M = max(8, int(4 * K * np.ceil(np.log2(K + 1))))

    full = recover_coefficients_pauli_cs(
        state_prep_circuit,
        support_indices=support,
        num_measurements=default_M,
        shots=shots,
        seed=seed,
    )
    return np.asarray([full[idx] for idx in support], dtype=complex)


def _validate_support_indices(support_indices: Sequence[int], num_qubits: int) -> list[int]:
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    dim = 1 << int(num_qubits)
    if not support_indices:
        raise ValueError("support_indices must be non-empty")

    support = [int(i) for i in support_indices]
    if len(set(support)) != len(support):
        raise ValueError("support_indices must be unique")
    for idx in support:
        if idx < 0 or idx >= dim:
            raise ValueError(f"support index {idx} out of range [0, {dim})")
    return support


def _validate_pauli_string(pauli: str, num_qubits: int) -> None:
    if len(pauli) != num_qubits:
        raise ValueError("Pauli string length must equal num_qubits")
    allowed = {"I", "X", "Y", "Z"}
    bad = [c for c in pauli if c not in allowed]
    if bad:
        raise ValueError(f"Invalid Pauli character(s): {bad}")


def _resolve_simulator(backend, num_qubits: int) -> AerSimulator:
    if backend is None:
        noise_model = make_custom_noise_model(
            p_1q=4.239e-4,
            p_2q=3.416e-3,
            p_meas=1e-2,
            coherent_phase=0.0,
            n_qubits=max(10, int(num_qubits)),
        )
        return AerSimulator(
            noise_model=noise_model,
            method="density_matrix",
            device="CPU",
        )
        # return AerSimulator(device="CPU")
    if isinstance(backend, AerSimulator):
        return backend
    return AerSimulator.from_backend(backend)


def _build_pauli_measurement_circuit(state_prep_circuit, pauli: str) -> QuantumCircuit:
    n = int(state_prep_circuit.num_qubits)
    qc = QuantumCircuit(n)
    qc.compose(state_prep_circuit, qubits=range(n), inplace=True)

    for q in range(n):
        p = pauli[q]
        if p == "X":
            qc.h(q)
        elif p == "Y":
            qc.sdg(q)
            qc.h(q)

    qc.measure_all()
    return qc


def _expectation_from_probabilities(pauli: str, probs: np.ndarray) -> float:
    n = len(pauli)
    non_identity_qubits = [q for q, p in enumerate(pauli) if p != "I"]
    if not non_identity_qubits:
        return 1.0

    ex = 0.0
    for idx, p in enumerate(np.asarray(probs, dtype=float)):
        if p == 0.0:
            continue
        parity = 0
        for q in non_identity_qubits:
            parity ^= (idx >> q) & 1
        eig = 1.0 if parity == 0 else -1.0
        ex += eig * p
    return float(ex)


def _pauli_element_on_basis_pair(pauli: str, s_a: int, s_b: int, num_qubits: int) -> complex:
    """Compute <s_a|P|s_b> for computational basis indices s_a,s_b."""
    val = 1.0 + 0.0j
    for q in range(num_qubits):
        p = pauli[q]
        a = (s_a >> q) & 1
        b = (s_b >> q) & 1

        if p == "I":
            if a != b:
                return 0.0 + 0.0j
            continue

        if p == "Z":
            if a != b:
                return 0.0 + 0.0j
            val *= 1.0 if a == 0 else -1.0
            continue

        if p == "X":
            if a != (1 - b):
                return 0.0 + 0.0j
            continue

        # p == "Y"
        if a != (1 - b):
            return 0.0 + 0.0j
        # <0|Y|1> = -i, <1|Y|0> = +i
        if a == 0 and b == 1:
            val *= -1j
        else:
            val *= 1j

    return val


__all__ = [
    "recover_coefficients_pauli_cs",
    "measure_pauli_expectations",
    "restrict_pauli_to_support",
    "sample_random_paulis",
    "reconstruct_rho_projected_gradient",
    "project_to_hermitian",
    "project_to_psd_trace1",
    "project_to_rank1",
    "principal_eigenvector",
    "fix_global_phase",
    "phase2_recover_coefficients",
]
