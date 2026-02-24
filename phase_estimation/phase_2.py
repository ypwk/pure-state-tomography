from typing import Dict, Optional

import cvxpy as cp
import numpy as np
from qiskit import QuantumCircuit

import src.qutils as qutils

# ------------------------------------------------------------
# Helper: build tensor-product Pauli matrices
# ------------------------------------------------------------

_PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_from_string(s: str) -> np.ndarray:
    """Kronecker product Pauli matrix from a string like 'IXYZ'."""
    P = np.array([[1]], dtype=complex)
    for ch in reversed(s):  # <--- reverse to match Qiskit qubit order
        P = np.kron(P, _PAULI_1Q[ch])
    return P


# ------------------------------------------------------------
# Main reconstruction function
# ------------------------------------------------------------

def reconstruct_density_matrix(
        measurements: Dict[str, float],
        epsilon: Optional[float] = None,
        support_basis: Optional[np.ndarray] = None,
        promote_low_rank: bool = True,
        solver: str = "SCS",
):
    """
    Convex quantum state reconstruction following Gross et al. (2009).

    Parameters
    ----------
    measurements : dict
        Maps Pauli strings -> measured expectations tr(P rho).
        Pauli strings must all have the same length (number of qubits).
    epsilon : float or None
        If None: enforce exact equality constraints (noise-free).
        If float: enforce ||A(rho) - b||_2 <= epsilon (robust version).
    support_basis : ndarray or None
        If provided, an isometry V of shape (d, r0) whose columns span
        a known support or support superset. The optimization is carried
        out over X with rho = V X V†.
    promote_low_rank : bool
        If True, minimize nuclear norm (trace norm).
        If False, solve feasibility / least-squares only.
        This should be False when the exact support is known.
    solver : str
        CVXPY solver name.

    Returns
    -------
    rho_hat : ndarray
        Reconstructed density matrix in the full Hilbert space.
    """

    # --------------------------------------------------------
    # Parse dimensions and measurements
    # --------------------------------------------------------

    pauli_strings = list(measurements.keys())
    b = np.array([measurements[p] for p in pauli_strings], dtype=float)

    n = len(pauli_strings[0])  # number of qubits
    d = 2 ** n  # Hilbert space dimension

    P_list = [pauli_from_string(p) for p in pauli_strings]

    # --------------------------------------------------------
    # Decide parameterization: full space or reduced support
    # --------------------------------------------------------

    if support_basis is None:
        # Full d x d density matrix
        var_dim = d
        V = None
        X = cp.Variable((d, d), complex=True)

        # Map variable to physical density matrix
        rho_expr = X

    else:
        # Reduced r0 x r0 variable with rho = V X V†
        V = support_basis
        r0 = V.shape[1]
        var_dim = r0

        X = cp.Variable((r0, r0), complex=True)
        rho_expr = V @ X @ V.conj().T

    # --------------------------------------------------------
    # Base physical constraints
    # --------------------------------------------------------

    constraints = [
        X - X.H == 0,  # Hermitian
        X >> 0,  # PSD
        cp.trace(rho_expr) == 1
    ]

    # --------------------------------------------------------
    # Stack measurement operator A(rho)
    # --------------------------------------------------------

    meas_exprs = []
    for P in P_list:
        if V is None:
            meas_exprs.append(cp.real(cp.trace(P @ X)))
        else:
            P_eff = V.conj().T @ P @ V
            meas_exprs.append(cp.real(cp.trace(P_eff @ X)))

    A_rho = cp.vstack(meas_exprs)  # shape (m, 1)
    b_vec = b.reshape((-1, 1))

    if epsilon is None:
        constraints.append(A_rho == b_vec)
    else:
        constraints.append(cp.norm(A_rho - b_vec, 2) <= epsilon)

    # --------------------------------------------------------
    # Objective selection
    # --------------------------------------------------------

    if promote_low_rank:
        # Nuclear norm promotes low rank when support is unknown or overcomplete
        objective = cp.Minimize(cp.normNuc(X))
    else:
        # When exact support is known, nuclear norm is constant (trace = 1)
        # Use feasibility or least-squares fitting instead
        if epsilon is None:
            objective = cp.Minimize(0)
        else:
            objective = cp.Minimize(cp.norm(A_rho - b_vec, 2))

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    if X.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
        raise ValueError(
            "Density-matrix reconstruction failed; "
            f"status={problem.status}"
        )

    # --------------------------------------------------------
    # Return full density matrix
    # --------------------------------------------------------

    if V is None:
        return X.value
    else:
        return V @ X.value @ V.conj().T


# ------------------------------------------------------------
# Phase-2 helpers: Pauli measurements + compressed sensing
# ------------------------------------------------------------


def _basis_change_for_pauli(qc: QuantumCircuit, pauli: str) -> None:
    """In-place basis change so Z-measurement equals Pauli measurement."""
    for q, p in enumerate(pauli):
        if p == "X":
            qc.h(q)
        elif p == "Y":
            qc.sdg(q)
            qc.h(q)
        elif p == "Z" or p == "I":
            continue
        else:
            raise ValueError(f"Invalid Pauli character: {p}")


def _expectation_from_probs(pauli: str, probs: np.ndarray) -> float:
    """Compute <P> from Z-basis probabilities after basis change."""
    n = len(pauli)
    dim = 1 << n
    if probs.shape[-1] != dim:
        raise ValueError("Probability vector dimension mismatch.")

    indices = np.arange(dim, dtype=np.uint32)
    eigen = np.ones(dim, dtype=float)
    for q, p in enumerate(pauli):
        if p == "I":
            continue
        bit = (indices >> q) & 1
        eigen *= 1.0 - 2.0 * bit  # 0 -> +1, 1 -> -1
    return float(np.sum(eigen * probs))


def _sample_pauli_strings(
    n_qubits: int,
    num_measurements: int,
    rng: np.random.Generator,
) -> list[str]:
    """Deterministic core + randomized tail, without repeats."""
    if num_measurements <= 0:
        return []

    max_unique = 4 ** n_qubits
    num_measurements = min(num_measurements, max_unique)

    result: list[str] = []
    used: set[str] = set()

    # # # Core set
    # core_candidates = ["I" * n_qubits, "Z" * n_qubits, "X" * n_qubits, "Y" * n_qubits]
    # for s in core_candidates:
    #     if len(result) >= num_measurements:
    #         break
    #     if s not in used:
    #         result.append(s)
    #         used.add(s)

    # Random tail
    paulis = np.array(["I", "X", "Y", "Z"], dtype=object)
    while len(result) < num_measurements:
        s = "".join(rng.choice(paulis, size=n_qubits, replace=True))
        if s not in used:
            result.append(s)
            used.add(s)

    return result


def compressed_sensing_phase2_magnitudes(
        state_prep_circuit: QuantumCircuit,
        aer_sim,
        *,
        support: list[int],
        shots: int,
        num_measurements: int,
        seed: int | None = None,
        epsilon: float | None = None,
        solver: str = "SCS",
) -> np.ndarray:
    """
    Phase-2: estimate density matrix via Pauli measurements + convex recovery.
    Returns full Hilbert-space density matrix.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    if num_measurements <= 0:
        raise ValueError("num_measurements must be positive")

    n = state_prep_circuit.num_qubits
    d = 1 << n

    # Build support basis columns (computational basis vectors)
    support = sorted(set(int(i) for i in support))
    if not support:
        raise ValueError("Support is empty; phase-1 failed.")

    V = np.zeros((d, len(support)), dtype=complex)
    for j, idx in enumerate(support):
        V[idx, j] = 1.0

    rng = np.random.default_rng(seed)
    pauli_strings = _sample_pauli_strings(n, num_measurements, rng)

    measurements: Dict[str, float] = {}
    for p in pauli_strings:
        qc = QuantumCircuit(n)
        qc.compose(state_prep_circuit, inplace=True)
        _basis_change_for_pauli(qc, p)
        qc.measure_all()

        probs = qutils.run_circuit(
            aer_sim,
            qc,
            shots=shots,
            batch_size=1,
            base_seed=seed,
        )[0]
        measurements[p] = _expectation_from_probs(p, probs)

    if epsilon is None:
        # Conservative L2 noise bound across m measurements
        m = len(pauli_strings)
        epsilon = 5.0 * np.sqrt(m) / np.sqrt(shots)

    rho_hat = reconstruct_density_matrix(
        measurements=measurements,
        epsilon=epsilon,
        support_basis=V,
        promote_low_rank=False,
        solver=solver,
    )
    return rho_hat
