#!/usr/bin/env python
"""Run phase-estimation tomography on GHZ chains of increasing size."""

from __future__ import annotations

import argparse
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from phase_estimation.algorithm import reconstruct_k_sparse_state


def build_ghz_circuit(num_qubits: int) -> QuantumCircuit:
    if num_qubits < 2:
        raise ValueError("GHZ requires at least 2 qubits")
    qc = QuantumCircuit(num_qubits, name=f"ghz_{num_qubits}")
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def fidelity_up_to_global_phase(psi: np.ndarray, phi: np.ndarray) -> float:
    idx = int(np.argmax(np.abs(phi)))
    phase = np.angle(psi[idx]) - np.angle(phi[idx])
    aligned = psi * np.exp(-1j * phase)
    return float(np.abs(np.vdot(phi, aligned)) ** 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phase-estimation tomography on GHZ chains."
    )
    parser.add_argument("--min-n", type=int, default=2, help="Minimum GHZ size.")
    parser.add_argument("--max-n", type=int, default=5, help="Maximum GHZ size.")
    parser.add_argument(
        "--phase1-shots", type=int, default=4096, help="Shots for support discovery."
    )
    parser.add_argument(
        "--phase2-shots", type=int, default=8192, help="Shots for LS recovery."
    )
    parser.add_argument(
        "--phase-bits",
        type=int,
        default=None,
        help="Phase-estimation register bits (default: algorithm heuristic).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_n < 2:
        raise ValueError("--min-n must be >= 2")
    if args.max_n < args.min_n:
        raise ValueError("--max-n must be >= --min-n")

    print(
        "n | phase_bits | support_size | support | fidelity | elapsed_s",
        flush=True,
    )
    print("-" * 80, flush=True)

    for n in range(args.min_n, args.max_n + 1):
        ghz = build_ghz_circuit(n)
        target = Statevector.from_instruction(ghz).data

        t0 = time.perf_counter()
        result = reconstruct_k_sparse_state(
            ghz,
            phase_register_bits=args.phase_bits,
            phase1_shots=args.phase1_shots,
            phase2_shots=args.phase2_shots,
        )
        elapsed = time.perf_counter() - t0

        fid = fidelity_up_to_global_phase(result.statevector, target)
        support_bits = [format(i, f"0{n}b") for i in result.support_indices]

        print(
            f"{n:>1} | {result.phase_register_bits:>10} | {len(result.support_indices):>12} | "
            f"{support_bits} | {fid:>8.6f} | {elapsed:>9.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
