#!/usr/bin/env python3
"""
Sanity check: verify Pauli ordering between Qiskit measurement
and pauli_from_string in phase_2.py.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from phase_estimation.phase_2 import pauli_from_string


def _counts_to_probs(counts: dict[str, int], n_qubits: int, shots: int) -> np.ndarray:
    dim = 1 << n_qubits
    probs = np.zeros(dim, dtype=float)
    for bitstr, c in counts.items():
        probs[int(bitstr, 2)] = c / shots
    return probs


def _expectation_from_probs(pauli: str, probs: np.ndarray) -> float:
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


def build_state_01() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.x(0)  # |q1 q0> = |0 1>
    return qc


def run_sanity_check(shots: int = 2000, seed: int = 123) -> None:
    qc = build_state_01()
    qc_meas = qc.copy()
    qc_meas.measure_all()

    sim = AerSimulator(method="density_matrix")
    result = sim.run(qc_meas, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts(qc_meas)
    probs = _counts_to_probs(counts, n_qubits=2, shots=shots)

    # Expected: Z on q0 gives -1 for |01>, Z on q1 gives +1
    exp_zi_meas = _expectation_from_probs("ZI", probs)
    exp_iz_meas = _expectation_from_probs("IZ", probs)

    # Compare with operator expectations from pauli_from_string
    qc_dm = qc.copy()
    qc_dm.save_density_matrix()
    rho = np.asarray(sim.run(qc_dm, seed_simulator=seed).result().data(0)["density_matrix"])
    exp_zi_op = float(np.real(np.trace(pauli_from_string("ZI") @ rho)))
    exp_iz_op = float(np.real(np.trace(pauli_from_string("IZ") @ rho)))

    print("State |01> sanity check")
    print(f"Measured <ZI> (q0) = {exp_zi_meas:.6f} (expected -1)")
    print(f"Measured <IZ> (q1) = {exp_iz_meas:.6f} (expected +1)")
    print(f"Operator <ZI>      = {exp_zi_op:.6f}")
    print(f"Operator <IZ>      = {exp_iz_op:.6f}")


def main() -> None:
    run_sanity_check()


if __name__ == "__main__":
    main()
