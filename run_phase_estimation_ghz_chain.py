#!/usr/bin/env python
"""Run phase-estimation tomography on GHZ chains of increasing size."""

from __future__ import annotations

import argparse
import csv
import os
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
    parser.add_argument("--max-n", type=int, default=10, help="Maximum GHZ size.")
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of repeated runs per GHZ size."
    )
    parser.add_argument(
        "--phase1-shots", type=int, default=4096, help="Shots for support discovery."
    )
    parser.add_argument(
        "--phase2-shots", type=int, default=4096, help="Shots for LS recovery."
    )
    parser.add_argument(
        "--phase-bits",
        type=int,
        default=2,
        help="Phase-estimation register bits (default: algorithm heuristic).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="ghz_phase_estimation_metrics.csv",
        help="Path to CSV file for per-run datapoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional master RNG seed for reproducible, non-identical runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_n < 2:
        raise ValueError("--min-n must be >= 2")
    if args.max_n < args.min_n:
        raise ValueError("--max-n must be >= --min-n")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")

    rng = np.random.default_rng(args.seed)

    print(
        "n | runs | avg_fid | std_fid | avg_elapsed_s | support_hit_rate",
        flush=True,
    )
    print("-" * 100, flush=True)
    fieldnames = [
        "n",
        "run_idx",
        "phase1_shots",
        "phase2_shots",
        "fidelity",
        "elapsed_s",
        "support_size",
        "support_hit",
        "support_indices",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for n in range(args.min_n, args.max_n + 1):
            ghz = build_ghz_circuit(n)
            target = Statevector.from_instruction(ghz).data
            expected_support = {0, (2**n) - 1}

            fidelities: list[float] = []
            elapsed_times: list[float] = []
            support_hits = 0
            phase_bits_used = None

            for run_idx in range(args.runs):
                run_seed = int(rng.integers(0, 2**31 - 1))
                t0 = time.perf_counter()
                result = reconstruct_k_sparse_state(
                    ghz,
                    phase_register_bits=args.phase_bits,
                    phase1_shots=args.phase1_shots,
                    phase2_shots=args.phase2_shots,
                    seed=run_seed,
                )
                elapsed = time.perf_counter() - t0

                fidelity = fidelity_up_to_global_phase(result, target)
                nonzero_indices = set(result.nonzero()[0])
                support_hit = int(nonzero_indices == expected_support)
                support_str = " ".join(format(i, f"0{n}b") for i in sorted(nonzero_indices))

                writer.writerow(
                    {
                        "n": n,
                        "run_idx": run_idx,
                        "phase1_shots": args.phase1_shots,
                        "phase2_shots": args.phase2_shots,
                        "fidelity": fidelity,
                        "elapsed_s": elapsed,
                        "support_size": len(nonzero_indices),
                        "support_hit": support_hit,
                        "support_indices": support_str,
                    }
                )
                csv_file.flush()
                os.fsync(csv_file.fileno())

                fidelities.append(fidelity)
                elapsed_times.append(elapsed)
                support_hits += support_hit

            print(
                f"{n:>1} | {args.runs:>4} | | "
                f"{np.mean(fidelities):>7.6f} | {np.std(fidelities):>7.6f} | "
                f"{np.mean(elapsed_times):>13.4f} | {support_hits / args.runs:>16.4f}",
                flush=True,
            )
    print(f"\nSaved per-run datapoints to: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
