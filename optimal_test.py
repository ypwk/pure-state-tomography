import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity
from qiskit_aer import AerSimulator

from src.noise_modeling import make_custom_noise_model


def pauli_measurement_circuit(base_circuit, pauli_string):
    """
    pauli_string: e.g. 'XZY'
    """
    n = len(pauli_string)
    qc = base_circuit.remove_final_measurements(inplace=False)

    for i, p in enumerate(pauli_string):
        if p == "X":
            qc.h(i)
        elif p == "Y":
            qc.sdg(i)
            qc.h(i)
        # Z: do nothing

    qc.measure_all()
    return qc


def pauli_strings(n):
    return ["".join(p) for p in itertools.product("IXYZ", repeat=n)]


# single-qubit Pauli matrices
_PAULI_SINGLE = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_matrix_from_string(pstr):
    """
    Construct a tensor-product Pauli matrix from a string like 'IXYZ'.

    Parameters
    ----------
    pstr : str
        Pauli string over alphabet {'I','X','Y','Z'}.

    Returns
    -------
    np.ndarray
        2^n × 2^n Pauli matrix with eigenvalues ±1.
    """
    mat = _PAULI_SINGLE[pstr[0]]
    for p in pstr[1:]:
        mat = np.kron(mat, _PAULI_SINGLE[p])
    return mat


def pauli_mean_from_counts(counts, pstr):
    """
    Estimate N_j = average ±1 outcome for a Pauli string measurement.
    Assumes the circuit maps measurement of pstr into computational Z-basis bits.

    counts: dict like {"010": 123, ...}
    pstr:   e.g. "IXYZ" with length n (qubit 0 corresponds to pstr[0])
    """
    shots = sum(counts.values())
    n = len(pstr)

    mean = 0.0
    for bitstring, c in counts.items():
        # Qiskit: rightmost bit is qubit 0
        bits = bitstring[::-1]

        eig = 1
        for q in range(n):
            if pstr[q] == "I":
                continue
            eig *= 1 if bits[q] == "0" else -1

        mean += eig * c

    return mean / shots


def adaptive_threshold(mean, shots, d, eps=1e-12):
    return np.sqrt(
        4 * (1 - mean ** 2 + eps) * np.log(d) / shots
    )


def run_thresholded_tomography(
        ghz_circuit,
        sim,
        shots=4096,
        threshold_type="hard"
):
    """
    Implements the Pauli-thresholded density matrix estimator
    of Cai et al. (2016).

    Parameters
    ----------
    ghz_circuit : QuantumCircuit
        State preparation circuit.
    sim : Backend
        Qiskit simulator backend.
    shots : int
        Number of measurements per Pauli observable.
    threshold_type : str
        "hard" or "soft" thresholding.

    Returns
    -------
    rho_hat : np.ndarray
        Estimated density matrix.
    beta_hat : dict
        Thresholded Pauli coefficients.
    """

    n = ghz_circuit.num_qubits
    d = 2 ** n
    p = d ** 2

    # universal threshold λ = sqrt(4 log d / n)
    # lam = np.sqrt(4 * np.log(d) / shots)

    beta_hat = {}

    for pstr in pauli_strings(n):
        if pstr == "I" * n:
            continue  # identity handled separately

        qc = pauli_measurement_circuit(ghz_circuit, pstr)
        tqc = transpile(qc, sim, optimization_level=0)
        counts = sim.run(tqc, shots=shots).result().get_counts()

        # map bitstrings → ±1 outcomes
        mean = pauli_mean_from_counts(counts, pstr)

        # thresholding step (Eq. 2.6)
        lam_j = adaptive_threshold(mean, shots, d)

        if threshold_type == "hard":
            beta = mean if abs(mean) >= lam_j else 0.0
        elif threshold_type == "soft":
            beta = np.sign(mean) * max(abs(mean) - lam_j, 0.0)
        else:
            raise ValueError("threshold_type must be 'hard' or 'soft'")

        beta_hat[pstr] = beta

    # reconstruct density matrix (Eq. 2.7)
    rho_hat = np.eye(d, dtype=complex) / d
    for pstr, beta in beta_hat.items():
        B = pauli_matrix_from_string(pstr)  # ±1 Pauli tensor
        rho_hat += (beta / d) * B

    return rho_hat, beta_hat


def pauli_expectation(counts, pauli_string):
    shots = sum(counts.values())
    exp = 0.0

    for bitstring, c in counts.items():
        parity = 1
        for b, p in zip(bitstring[::-1], pauli_string):
            if p in "XYZ" and b == "1":
                parity *= -1
        exp += parity * c / shots

    return exp


def ghz_circuit(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def ideal_ghz_density_matrix(n: int) -> DensityMatrix:
    dim = 2 ** n
    ghz = np.zeros(dim, dtype=complex)
    ghz[0] = 1.0
    ghz[-1] = 1.0
    ghz /= np.sqrt(2)
    return DensityMatrix(np.outer(ghz, ghz.conj()))


def to_qiskit_density_matrix(rho):
    # enforce Hermiticity
    rho = 0.5 * (rho + rho.conj().T)

    # enforce unit trace
    rho = rho / np.trace(rho)

    return DensityMatrix(rho)


def project_to_psd_density_matrix(rho, eps=0.0):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, eps, None)
    rho_psd = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    rho_psd /= np.trace(rho_psd)
    return DensityMatrix(rho_psd)


NOISE = False
N_REPEATS = 16
SHOTS = 4096

if NOISE:
    noise_model = make_custom_noise_model(
        p_1q=4.239e-4,
        p_2q=3.416e-3,
        p_meas=1e-2,
        coherent_phase=0.0,
        n_qubits=10,
    )
    sim = AerSimulator(method="density_matrix", noise_model=noise_model, device="GPU")
else:
    sim = AerSimulator(method="density_matrix", device="GPU")

results = {}

import time

for n in range(2, 7):
    start_n = time.perf_counter()

    fidelities = []
    rho_ideal = ideal_ghz_density_matrix(n)

    for _ in range(N_REPEATS):
        rho_hat, beta_hat = run_thresholded_tomography(
            ghz_circuit(n),
            sim,
            shots=SHOTS
        )
        rho_dm = project_to_psd_density_matrix(rho_hat)
        fidelity = state_fidelity(rho_dm, rho_ideal)
        fidelities.append(fidelity)

    elapsed_n = time.perf_counter() - start_n

    results[n] = fidelities
    print(f"n: {n} | mean fidelity: {np.mean(fidelities)} | time: {elapsed_n:.3f}s")
    print(fidelities)
