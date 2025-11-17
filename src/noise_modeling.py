from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError
from qiskit_aer.noise.errors import depolarizing_error
from qiskit.quantum_info import Pauli
import numpy as np


def make_custom_noise_model(
    p_1q: float = 0.0,
    p_2q: float = 0.0,
    p_meas: float = 0.0,
    coherent_phase: float = 0.0,
    n_qubits: int = 5,
) -> NoiseModel:
    """
    Create a customizable Aer NoiseModel.

    Args:
        p_1q : depolarizing probability for 1Q gates
        p_2q : depolarizing probability for 2Q gates
        p_meas : readout (bit-flip) probability
        coherent_phase : small phase rotation injected after 1Q/H gates
        n_qubits : size of the device

    Returns:
        NoiseModel
    """
    nm = NoiseModel()

    # -- Noise channels --

    # Depolarizing channels
    c_1q = depolarizing_error(p_1q, 1) if p_1q > 0 else None
    c_2q = depolarizing_error(p_2q, 2) if p_2q > 0 else None

    # Measurement bit-flip noise
    meas_flip = (
        ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
        if p_meas > 0 else None
    )

    # Small coherent Z-rotation (e.g. calibration drift)
    coh_phase = (
        QuantumError([(np.exp(1j * coherent_phase *
                      Pauli('Z').to_matrix()), 1.0)])
        if coherent_phase != 0 else None
    )

    # -- Apply noise to gate types --

    target_1q = ['h', 'u1', 'u2', 'u3', 'rz', 'sx', 'x', 'id']

    for q in range(n_qubits):
        if c_1q:
            nm.add_quantum_error(c_1q, target_1q, [q])
        if coh_phase:
            nm.add_quantum_error(coh_phase, ['h'], [q])  # only after Hadamard

        if meas_flip:
            nm.add_readout_error(meas_flip, [q])

    if c_2q:
        for q0 in range(n_qubits):
            for q1 in range(n_qubits):
                if q0 != q1:
                    nm.add_quantum_error(c_2q, ['cx'], [q0, q1])

    return nm
