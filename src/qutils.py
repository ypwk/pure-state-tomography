"""
This file contains utility code for quantum-related functions.

The utility functions provided include:
- infer_target: Calculates and returns the value of an entry using previously inferred
    values in the measurement results.
- create_circuit: Initializes a state as a qiskit QuantumCircuit.
- run_circuit: Runs the circuit on a simulator or a real quantum device.
- find_nonzero_positions: Finds positions with nonzero counts in an np.array.

Additionally, the file defines an enumeration for measurement types and imports
necessary modules and classes:
- m_type: An enumeration for measurement types, including real Hadamard, complex
    Hadamard, and identity.

See each function's respective docstring for detailed usage and parameter information.
"""

import numpy as np

import os
from typing import Optional

from qiskit import QuantumCircuit, transpile, result
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

from enum import Enum

EPSILON = 5e-2


class m_type(Enum):
    real_hadamard = 1
    cmplx_hadamard = 2
    identity = 3


class execution_type(Enum):
    ibm_qpu = 1
    simulator = 2
    statevector = 3


class tomography_type(Enum):
    process = 1
    state = 2


def find_nonzero_positions(counts, epsilon=EPSILON) -> list:
    """Finds positions with nonzero counts in the counts np.array

    Args:
        counts (numpy.np.ndarray): np.array filled with counts

    Returns:
        List: a list of counts
    """
    positions = []
    for c in range(len(counts)):
        if counts[c] > epsilon:
            positions.append(c)
    return positions


def infer_target(target_idx, source_idx, source_val, h_measure, v_measure) -> np.ndarray:
    """Calculates and returns the value of an entry using previously inferred values in
    the measurement results.

    Args:
        target_idx (int): The index of the target value to infer
        source_idx (int): The index of the value to use to infer the target
        source_val (numpy.np.ndarray): The source value
        h_measure (numpy.np.ndarray): The np.array of measurements with the Hadamard gate
        v_measure (numpy.np.ndarray): The np.array of measurements with the alternate gate

    Returns: numpy.np.ndarray
    """

    res = np.array([0.0, 0.0])
    if target_idx < source_idx:  # backwards
        res[0] = (
            source_val[1] * (v_measure[source_idx] - v_measure[target_idx])
            + source_val[0] * (h_measure[target_idx] - h_measure[source_idx])
        ) / (2 * (source_val[0] * source_val[0] + source_val[1] * source_val[1]))

        res[1] = (
            source_val[0] * (v_measure[source_idx] - v_measure[target_idx])
            + source_val[1] * (h_measure[target_idx] - h_measure[source_idx])
        ) / (2 * (source_val[0] * source_val[0] + source_val[1] * source_val[1]))

    else:  # forwards
        res[0] = (
            source_val[0] * (h_measure[source_idx] - h_measure[target_idx])
            - source_val[1] * (v_measure[target_idx] - v_measure[source_idx])
        ) / (2 * (source_val[0] * source_val[0] + source_val[1] * source_val[1]))

        res[1] = (
            source_val[0] * (v_measure[target_idx] - v_measure[source_idx])
            + source_val[1] * (h_measure[source_idx] - h_measure[target_idx])
        ) / (2 * (source_val[0] * source_val[0] + source_val[1] * source_val[1]))

    return res


def infer_block(target_idx, source_idx, source_val, h_measure, v_measure) -> np.ndarray:
    """Calculates and returns the value of an entry using previously inferred values in
    the measurement results.

    Args:
        target_idx (int): The index of the target value to
        
        
    """
    


def create_vector_circuit(state, n_qubits) -> QuantumCircuit:
    """Initializes a state as a qiskit QuantumCircuit

    Args:
        state (numpy.np.ndarray): The state to initialize
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
        state (numpy.np.ndarray): The state to initialize
        n_qubits (int): The number of qubits used to represent the staet

    Returns:
        qiskit.QuantumCircuit:
    """
    qc = QuantumCircuit(n_qubits)
    for a in range(n_qubits // 2):
        qc.h(n_qubits // 2 + a)
    for a in range(n_qubits // 2 - 1, -1, -1):
        qc.cx(n_qubits // 2 + a, a)
    qc.append(UnitaryGate(state), range(n_qubits // 2))
    return qc


def run_circuit(aer_sim, qc, shots=1024, backend=None) -> result.counts.Counts:
    """Runs the circuit on the simulator

    Args:
        qc (qiskit.QuantumCircuit): Quantum circuit to run
        shots (int): Number of shots to take
        backend: Backend device to mimic

    Returns:
        numpy.np.ndarray: An np.array of result counts
    """
    t_qc = transpile(qc, aer_sim, optimization_level=1)
    result = aer_sim.run(t_qc, shots=shots, device="GPU").result()
    return result.get_counts(qc)


def circuit_to_statevector(qc: QuantumCircuit) -> np.ndarray:
    """Converts a circuit into a statevector

    Args:
        qc (QuantumCircuit): The circuit to convert

    Returns:
        np.ndarray: The vector representation of the circuit
    """
    copied = qc.copy("execution")
    copied.save_statevector()
    simulator = AerSimulator(method="statevector")
    raw_result = simulator.run(copied).result()
    return np.asarray(raw_result.get_statevector(copied))


def circuit_to_unitary(qc: QuantumCircuit) -> np.ndarray:
    """Converts a circuit into a unitary matrix

    Args:
        qc (QuantumCircuit): The circuit to convert

    Returns:
        np.ndarray: The unitary matrix representation of the circuit
    """
    copied = qc.copy("execution")
    copied.save_unitary()
    simulator = AerSimulator(method="unitary")
    raw_result = simulator.run(copied).result()
    return np.asarray(raw_result.get_unitary(copied))


def save_qasm3(circ: QuantumCircuit, path: str) -> None:
    from qiskit.qasm3 import dumps
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps(circ))


def load_qasm3(path: str) -> QuantumCircuit:
    from qiskit.qasm3 import loads
    with open(path, "r", encoding="utf-8") as f:
        return loads(f.read())


def load_from_experiment_dir(exp_dir: str) -> Optional[QuantumCircuit]:
    """
    Loads circuit from `exp_dir` by priority:
      1) circuit.qpy
      2) circuit.qasm (OpenQASM 3)
    Returns None if nothing is found.
    """
    qasm = os.path.join(exp_dir, "circuit.qasm")
    if os.path.isfile(qasm):
        return load_qasm3(qasm)
    return None


def calculate_fidelity(ideal, actual, type):
    """
    Calculates the fidelity between the ideal and actual quantum states.

    Parameters:
        ideal (np.ndarray): The ideal quantum state.
        actual (np.ndarray): The actual quantum state.
        type: type of fidelity to calculate, process or state

    Returns:
        float: The fidelity value.
    """
    if type == tomography_type.process:
        inp_dim = ideal.shape[0]
        ideal = np.reshape(ideal, (inp_dim * inp_dim,))
        actual = np.reshape(actual, (inp_dim * inp_dim,)).T
        inner_product = np.vdot(ideal, actual)
        return np.abs(inner_product) / (inp_dim)
    elif type == tomography_type.state:
        inner_product = np.vdot(ideal, actual)
        return np.abs(inner_product)


__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
