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
import time

import numpy as np

import os
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

from enum import Enum

import src.putils as putils

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
    """
    Vectorized version: operates on batches.
    Args:
        target_idx (int): index of target basis state
        source_idx (int): index of source basis state
        source_val (np.ndarray): (batch_size, 2) array of [Re, Im] values for the source
        h_measure (np.ndarray): (batch_size, dim) array of Hadamard measurement probs
        v_measure (np.ndarray): (batch_size, dim) array of complex-Hadamard measurement probs
    Returns:
        np.ndarray of shape (batch_size, 2)
    """
    # Denominator: (batch,)
    denom = 2 * (source_val[:, 0] ** 2 + source_val[:, 1] ** 2)

    if target_idx < source_idx:  # backwards
        re = (
            source_val[:, 1] * (v_measure[:, source_idx] -
                                v_measure[:, target_idx])
            + source_val[:, 0] *
            (h_measure[:, target_idx] - h_measure[:, source_idx])
        ) / denom
        im = (
            source_val[:, 0] * (v_measure[:, source_idx] -
                                v_measure[:, target_idx])
            + source_val[:, 1] *
            (h_measure[:, target_idx] - h_measure[:, source_idx])
        ) / denom
    else:  # forwards
        re = (
            source_val[:, 0] * (h_measure[:, source_idx] -
                                h_measure[:, target_idx])
            - source_val[:, 1] *
            (v_measure[:, target_idx] - v_measure[:, source_idx])
        ) / denom
        im = (
            source_val[:, 0] * (v_measure[:, target_idx] -
                                v_measure[:, source_idx])
            + source_val[:, 1] *
            (h_measure[:, source_idx] - h_measure[:, target_idx])
        ) / denom

    return np.stack([re, im], axis=1)  # (batch_size, 2)


def infer_partially_mixed_target(target_idx, source_idx, source_val, hadamards, op_pos, h_measure, v_measure) -> np.ndarray:
    """
    Vectorized version for partial mixing case.
    Args:
        target_idx (int)
        # source_idx (int)
        source_val (np.ndarray): (batch_size, 2)
        h_measure (np.ndarray): (batch_size, dim)
        v_measure (np.ndarray): (batch_size, dim)
    Returns:
        np.ndarray of shape (batch_size, 2)
    """
    all_hadamards = set(hadamards) | {op_pos}
    # need to scale to account for subspace scaling when mixed
    mixed_subspace_extra_size = putils.fast_pow(2, len(all_hadamards) - 1)
    denom = 4 * (source_val[:, 0] ** 2 + source_val[:, 1] ** 2)
    num_qubits = putils.fast_log2(h_measure.shape[1])
    non_hadamards = [i for i in range(num_qubits) if i not in all_hadamards]

    # Build mask of those non-hadamard bits
    block_mask = 0
    for i in non_hadamards:
        block_mask |= 1 << (num_qubits - 1 - i)

    # Block bitstring = bits of target_idx restricted to non-hadamards
    block_bitstring = target_idx & block_mask

    batch = h_measure.shape[0]  # 128

    had_list = sorted(all_hadamards)
    k = len(had_list)
    block_size = 1 << k  # 2^k

    h_block_values = np.zeros((batch, block_size))
    v_block_values = np.zeros((batch, block_size))

    for bits in range(block_size):
        idx = block_bitstring
        for j, q in enumerate(had_list):
            if (bits >> j) & 1:
                idx |= (1 << (num_qubits - 1 - q))

        h_block_values[:, bits] = h_measure[:, idx]
        v_block_values[:, bits] = v_measure[:, idx]

    # parity pattern for this block ordering
    s = np.arange(block_size)
    parity = np.vectorize(lambda x: x.bit_count() & 1)(s)  # 0 or 1
    signs = 1 - 2 * parity  # even → +1, odd → -1

    pos_idx = np.where(signs == +1)[0]
    neg_idx = np.where(signs == -1)[0]

    # print(f"pos idx {pos_idx}")
    # print(f"pos idx {neg_idx}")

    re_vals = []
    im_vals = []

    for p in pos_idx:
        for q in neg_idx:
            h_s = 2 * h_block_values[:, p]
            h_t = 2 * h_block_values[:, q]
            v_s = 2 * v_block_values[:, p]
            v_t = 2 * v_block_values[:, q]

            re = (
                source_val[:, 0] * (h_s - h_t)
                - source_val[:, 1] * (v_t - v_s)
            ) / denom * mixed_subspace_extra_size

            im = (
                source_val[:, 0] * (v_t - v_s)
                + source_val[:, 1] * (h_s - h_t)
            ) / denom * mixed_subspace_extra_size

            re_vals.append(re)
            im_vals.append(im)

            # print(re_vals, im_vals)

    # Average over all pos-neg combinations
    re_mean = np.mean(np.stack(re_vals, axis=0), axis=0)
    im_mean = np.mean(np.stack(im_vals, axis=0), axis=0)

    return np.stack([re_mean, im_mean], axis=1)


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


def run_circuit(aer_sim, qc, shots=1024, batch_size: int = 1) -> np.ndarray:
    """Runs the circuit on the simulator and prints runtime info + transpiled circuit structure."""

    dim = 1 << qc.num_qubits
    total_shots = shots

    def _counts_to_prob(counts):
        res = np.zeros(dim)
        for bitstr, c in counts.items():
            res[int(bitstr, 2)] = c / shots
        return res

    # Transpile and time execution
    # start_time = time.perf_counter()

    t_qc = transpile(qc, aer_sim, optimization_level=0,
                     layout_method="noise_adaptive")

    # # ----------------------------------------------------------------------
    # print("\n========== Transpiled Circuit ==========")
    # print(t_qc)

    # # Also print gate counts for quick comparison
    # gcounts = t_qc.count_ops()
    # print("Gate counts:", gcounts)

    # # Optional: print depth & number of 2-qubit gates
    # depth = t_qc.depth()
    # num_cx = gcounts.get('cx', 0) or gcounts.get('CX', 0) or gcounts.get("cz", 0) or gcounts.get("CZ", 0)
    # print(f"Circuit depth: {depth}, 2Q gates: {num_cx}")

    # print("========================================\n")
    # # ----------------------------------------------------------------------

    results = np.zeros((batch_size, dim))

    for i in range(batch_size):
        job = aer_sim.run(t_qc, shots=total_shots,
                          seed_simulator=np.random.randint(1e9))
        counts = job.result().get_counts(t_qc)
        results[i] = _counts_to_prob(counts)

    # end_time = time.perf_counter()
    # elapsed = end_time - start_time

    # print(f"[run_circuit] Time elapsed: {elapsed:.6f} seconds "
    #       f"({elapsed / batch_size:.6f} per batch)")

    return results


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
