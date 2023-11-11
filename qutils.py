"""
This file contains utility code for quantum-related functions.

The utility functions provided include:
- infer_target: Calculates and returns the value of an entry using previously inferred
    values in the measurement results.
- create_circuit: Initializes a state as a qiskit QuantumCircuit.
- run_circuit: Runs the circuit on a simulator or a real quantum device.
- find_nonzero_positions: Finds positions with nonzero counts in an array.

Additionally, the file defines an enumeration for measurement types and imports
necessary modules and classes:
- m_type: An enumeration for measurement types, including real Hadamard, complex
    Hadamard, and identity.

See each function's respective docstring for detailed usage and parameter information.
"""

from numpy import ndarray, array

from qiskit import QuantumCircuit, transpile, result
from qiskit.extensions import UnitaryGate
from qiskit_aer import AerSimulator

from enum import Enum


# EPSILON = 1.5e-2  # matrix
# EPSILON = 5e-2  # vector
EPSILON = 5e-5  # statevector


class m_type(Enum):
    real_hadamard = 1
    cmplx_hadamard = 2
    identity = 3


class execution_type(Enum):
    ibm_qpu = 1
    simulator = 2
    statevector = 3


def find_nonzero_positions(counts, epsilon=EPSILON) -> list:
    """Finds positions with nonzero counts in the counts array

    Args:
        counts (numpy.ndarray): array filled with counts

    Returns:
        List: a list of counts
    """
    positions = []
    for c in range(len(counts)):
        if counts[c] > epsilon:
            positions.append(c)
    return positions


def infer_target(target_idx, source_idx, source_val, h_measure, v_measure) -> ndarray:
    """Calculates and returns the value of an entry using previously inferred values in
    the measurement results.

    Args:
        target_idx (int): The index of the target value to infer
        source_idx (int): The index of the value to use to infer the target
        source_val (numpy.ndarray): The source value
        h_measure (numpy.ndarray): The array of measurements with the Hadamard gate
        v_measure (numpy.ndarray): The array of measurements with the alternate gate

    Returns: numpy.ndarray
    """

    res = array([0.0, 0.0])
    if target_idx < source_idx:  # backwards
        res[0] = (
            source_val[1] * (v_measure[source_idx] - v_measure[target_idx])
            + source_val[0] * (h_measure[target_idx] - h_measure[source_idx])
        ) / (2 * (source_val[0] * source_val[0] + source_val[1] * source_val[1]))

        res[1] = (
            source_val[0] * (v_measure[source_idx] - v_measure[target_idx])
            - source_val[1] * (h_measure[target_idx] - h_measure[source_idx])
        ) / (2 * (source_val[0] * source_val[0] + source_val[1] * source_val[1]))

    else:  # forwards
        res[0] = (
            source_val[1] * (v_measure[target_idx] - v_measure[source_idx])
            + source_val[0] * (h_measure[source_idx] - h_measure[target_idx])
        ) / (2 * (source_val[0] * source_val[0] - source_val[1] * source_val[1]))

        res[1] = (
            source_val[1] * (h_measure[source_idx] - h_measure[target_idx])
            + source_val[0] * (v_measure[target_idx] - v_measure[source_idx])
        ) / (2 * (source_val[0] * source_val[0] + source_val[1] * source_val[1]))

    return res


def create_vector_circuit(state, n_qubits) -> QuantumCircuit:
    """Initializes a state as a qiskit QuantumCircuit

    Args:
        state (numpy.ndarray): The state to initialize
        n_qubits (int): The number of qubits used to represent the staet

    Returns:
        qiskit.QuantumCircuit:
    """
    qc = QuantumCircuit(n_qubits)
    qc.initialize(state, [_ for _ in range(n_qubits)])
    return qc


def create_matrix_circuit(state, n_qubits) -> QuantumCircuit:
    """Initializes a state as a qiskit QuantumCircuit

    Args:
        state (numpy.ndarray): The state to initialize
        n_qubits (int): The number of qubits used to represent the staet

    Returns:
        qiskit.QuantumCircuit:
    """
    qc = QuantumCircuit(n_qubits)
    for a in range(n_qubits // 2):
        qc.h(n_qubits // 2 + a)
    for a in range(n_qubits // 2 - 1, -1, -1):
        qc.cnot(n_qubits // 2 + a, a)
    qc.append(UnitaryGate(state), range(n_qubits // 2))
    return qc


def run_circuit(qc, shots=1024, backend=None) -> result.counts.Counts:
    """Runs the circuit on the simulator

    Args:
        qc (qiskit.QuantumCircuit): Quantum circuit to run
        shots (int): Number of shots to take
        backend: Backend device to mimic

    Returns:
        numpy.ndarray: An array of result counts
    """
    aer_sim = AerSimulator.from_backend(backend)
    t_qc = transpile(qc, aer_sim)
    result = aer_sim.run(t_qc, shots=shots).result()
    return result.get_counts(qc)


__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
