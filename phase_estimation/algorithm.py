#!/usr/bin/env python
"""K-sparse pure-state tomography with phase estimation (small qubit counts).

This module implements a practical two-phase workflow inspired by
Gulbahar (2021), "K-sparse Pure State Tomography with Phase Estimation":

1) Discover support basis states with phase estimation over the Section IV
   circuit construction U_phi.
2) Recover complex coefficients on that support by linear least squares over
   measurement probabilities from local Pauli-basis settings.
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_aer import AerSimulator

from phase_estimation.phase_1 import phase1_recover_support
from phase_estimation.phase_2 import compressed_sensing_phase2_magnitudes
from src.noise_modeling import make_custom_noise_model


def project_density_matrix_to_pure_state(
        rho: DensityMatrix,
        *,
        atol: float = 1e-12,
) -> Statevector:
    """
    Project a (nearly pure) density matrix onto a pure state vector by
    taking the eigenvector corresponding to the largest eigenvalue.

    This is the optimal rank-1 approximation in fidelity.

    Parameters
    ----------
    rho : DensityMatrix
        Full density matrix (d x d), assumed Hermitian and trace 1.
    atol : float
        Numerical tolerance for eigenvalue clipping.

    Returns
    -------
    Statevector
        Normalized pure state |psi> approximating rho.
    """
    # Convert to NumPy array
    mat = np.asarray(rho.data, dtype=complex)

    # Hermitize defensively
    mat = 0.5 * (mat + mat.conj().T)

    # Eigen-decomposition (Hermitian → eigh)
    evals, evecs = np.linalg.eigh(mat)

    # Select dominant eigenpair
    idx = np.argmax(evals)
    lam = evals[idx]
    psi = evecs[:, idx]

    if lam < atol:
        raise ValueError(
            "Density matrix has no dominant eigenvalue; "
            "state is not close to pure."
        )

    # Normalize (eigh returns normalized vectors, but be explicit)
    psi = psi / np.linalg.norm(psi)

    return Statevector(psi)


def reconstruct_k_sparse_state(
        state_prep_circuit,
        *,
        phase_register_bits: int,
        phase1_shots: int,
        phase2_shots: int,
        phase2_num_measurements: int | None = None,
        seed: int | None = None,
):
    """
    Reconstruct a K-sparse pure n-qubit state using:
      Phase 1: support recovery via phase estimation
      Phase 2: coefficient recovery via compressed sensing

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (psi_hat, rho_hat) where psi_hat is the projected pure-state estimate
        and rho_hat is the reconstructed density matrix estimate.
    """
    # Phase 1: recover computational-basis support
    noise_model = make_custom_noise_model( # dont get rid of me!
        p_1q=4.239e-4,
        p_2q=3.416e-3,
        p_meas=1e-2,
        coherent_phase=0.0,
        n_qubits=10,
    )
    aer_sim = AerSimulator(
        noise_model=noise_model, # note that noise is disabled right now for testing. I will renable it later
        method="density_matrix",
        device="GPU",
    )
    # aer_sim = AerSimulator.from_backend(FakeMarrakesh())
    support = phase1_recover_support(
        state_prep_circuit,
        aer_sim,
        phase_register_bits=phase_register_bits,
        shots=phase1_shots,
        seed=seed,
    )

    # Phase 2: estimate full state from support-restricted Pauli CS
    if phase2_num_measurements is None:
        phase2_num_measurements = state_prep_circuit.num_qubits ** 2 * 2 ** (state_prep_circuit.num_qubits)

    rho_hat = compressed_sensing_phase2_magnitudes(
        state_prep_circuit,
        aer_sim,
        support=support,
        shots=phase2_shots,
        num_measurements=phase2_num_measurements,
        seed=seed,
    )
    psi_hat = project_density_matrix_to_pure_state(DensityMatrix(rho_hat))
    return psi_hat.data, rho_hat
