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

from phase_estimation.embed import embed_support_state
from phase_estimation.phase_1 import phase1_recover_support
from phase_estimation.phase_2 import recover_coefficients_pauli_cs



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
    """
    # Phase 1: recover computational-basis support
    support = phase1_recover_support(
        state_prep_circuit,
        phase_register_bits=phase_register_bits,
        shots=phase1_shots,
        seed=seed,
    )

    # Phase 2: estimate full state from support-restricted Pauli CS
    k = len(support)
    if phase2_num_measurements is None:
        phase2_num_measurements = max(8, int(4 * k * np.ceil(np.log2(k + 1))))

    full_state = recover_coefficients_pauli_cs(
        state_prep_circuit,
        support_indices=support,
        num_measurements=phase2_num_measurements,
        shots=phase2_shots,
        seed=seed,
    )

    # Optional consistency projection onto recovered support.
    # recover_coefficients_pauli_cs already returns a full vector with off-support 0.
    full_state = embed_support_state(
        [full_state[idx] for idx in support],
        support_indices=support,
        num_qubits=state_prep_circuit.num_qubits,
        normalize=True,
    )
    return full_state
