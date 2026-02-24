#!/usr/bin/env python
"""Run phase-estimation tomography on GHZ chains of increasing size."""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from phase_estimation.algorithm import reconstruct_k_sparse_state


def build_ghz_circuit(num_qubits: int) -> QuantumCircuit:
    if num_qubits < 2:
        raise ValueError("GHZ requires at least 2 qubits")
    qc = QuantumCircuit(num_qubits, name=f"ghz_{num_qubits}")
    qc.ry(np.pi / 3, 0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def fidelity_up_to_global_phase(psi: np.ndarray, phi: np.ndarray) -> float:
    idx = int(np.argmax(np.abs(phi)))
    phase = np.angle(psi[idx]) - np.angle(phi[idx])
    aligned = psi * np.exp(-1j * phase)
    return float(np.abs(np.vdot(phi, aligned)) ** 2)


def density_matrix_fidelity_to_pure_target(
    rho: np.ndarray, target_state: np.ndarray
) -> float:
    """For pure |psi>, fidelity F(rho, |psi><psi|) = <psi| rho |psi>."""
    value = np.vdot(target_state, rho @ target_state)
    return float(np.real(value))


def print_run_metadata(args: argparse.Namespace) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("RUN METADATA", flush=True)
    print(f"timestamp={timestamp}", flush=True)
    print(f"cwd={os.getcwd()}", flush=True)
    print(f"python={os.sys.version.split()[0]}", flush=True)
    print(f"numpy={np.__version__}", flush=True)
    print(f"qiskit={qiskit.__version__}", flush=True)
    print(
        "args="
        f"min_n={args.min_n} "
        f"max_n={args.max_n} "
        f"n_range={args.n_range} "
        f"runs={args.runs} "
        f"phase1_shots={args.phase1_shots} "
        f"phase2_shots={args.phase2_shots} "
        f"phase_bits={args.phase_bits} "
        f"seed={args.seed} "
        f"out_csv={args.out_csv}",
        flush=True,
    )
    print("-" * 80, flush=True)


def print_run_line(
    *,
    n: int,
    run_idx: int,
    run_seed: int,
    phase_bits: int,
    phase1_shots: int,
    phase2_shots: int,
    pure_fidelity: float,
    density_fidelity: float,
    elapsed: float,
    support_size: int,
    support_hit: int,
    support_indices: str,
) -> None:
    print(
        f"n={n} "
        f"run={run_idx:03d} "
        f"seed={run_seed} "
        f"phase_bits={phase_bits} "
        f"phase1_shots={phase1_shots} "
        f"phase2_shots={phase2_shots} "
        f"fid_pure={pure_fidelity:.6f} "
        f"fid_rho={density_fidelity:.6f} "
        f"elapsed_s={elapsed:.4f} "
        f"support_size={support_size} "
        f"support_hit={support_hit} "
        f"support=[{support_indices}]",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phase-estimation tomography on GHZ chains."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Single GHZ size (overrides --min-n/--max-n).",
    )
    parser.add_argument(
        "--n-range",
        type=str,
        default=None,
        help="Inclusive GHZ size range, e.g. '2:6' or '2-6' (overrides --min-n/--max-n unless --n is set).",
    )
    parser.add_argument("--min-n", type=int, default=3, help="Minimum GHZ size.")
    parser.add_argument("--max-n", type=int, default=10, help="Maximum GHZ size.")
    parser.add_argument(
        "--runs", type=int, default=128, help="Number of repeated runs per GHZ size."
    )
    parser.add_argument(
        "--phase1-shots", type=int, default=8196, help="Shots for support discovery."
    )
    parser.add_argument(
        "--phase2-shots", type=int, default=16384, help="Shots for LS recovery."
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
        default=None,
        help="Path to CSV file for per-run datapoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional master RNG seed for reproducible, non-identical runs.",
    )
    return parser.parse_args()


def _parse_n_range(n_range: str) -> tuple[int, int]:
    spec = n_range.strip()
    if ":" in spec:
        parts = spec.split(":")
    elif "-" in spec:
        parts = spec.split("-")
    else:
        raise ValueError("--n-range must use ':' or '-' separator, e.g. 2:6")
    if len(parts) != 2:
        raise ValueError("--n-range must have exactly two bounds, e.g. 2:6")
    lo = int(parts[0].strip())
    hi = int(parts[1].strip())
    return lo, hi


def main() -> None:
    args = parse_args()
    if args.n is not None:
        if args.n < 2:
            raise ValueError("--n must be >= 2")
        args.min_n = args.n
        args.max_n = args.n
    elif args.n_range is not None:
        lo, hi = _parse_n_range(args.n_range)
        args.min_n = lo
        args.max_n = hi
    if args.min_n < 2:
        raise ValueError("--min-n must be >= 2")
    if args.max_n < args.min_n:
        raise ValueError("--max-n must be >= --min-n")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join("experiments", "phase_bits_n_n_noise")
    os.makedirs(out_dir, exist_ok=True)
    if args.out_csv is None:
        if args.min_n == args.max_n:
            args.out_csv = os.path.join(
                out_dir, f"ghz_phase_estimation_metrics_n{args.min_n}.csv"
            )
        else:
            args.out_csv = os.path.join(out_dir, "ghz_phase_estimation_metrics.csv")
    print_run_metadata(args)

    # print(
    #     "n | runs | avg_fid | std_fid | avg_elapsed_s | support_hit_rate",
    #     flush=True,
    # )
    # print("-" * 100, flush=True)
    fieldnames = [
        "n",
        "run_idx",
        "phase1_shots",
        "phase2_shots",
        "fidelity_pure",
        "fidelity_density",
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

            pure_fidelities: list[float] = []
            density_fidelities: list[float] = []
            elapsed_times: list[float] = []
            support_hits = 0
            failed_attempts = 0
            run_idx = 0
            while run_idx < args.runs:
                run_seed = int(rng.integers(0, 2**31 - 1))
                try:
                    t0 = time.perf_counter()
                    result_pure, result_rho = reconstruct_k_sparse_state(
                        ghz,
                        phase_register_bits=args.phase_bits,
                        phase1_shots=args.phase1_shots,
                        phase2_shots=args.phase2_shots,
                        seed=run_seed,
                    )
                    elapsed = time.perf_counter() - t0
                except Exception as exc:
                    failed_attempts += 1
                    print(
                        f"\nERROR n={n} failed_attempt={failed_attempts} "
                        f"requested_run={run_idx + 1}/{args.runs} seed={run_seed} "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue

                pure_fidelity = fidelity_up_to_global_phase(result_pure, target)
                density_fidelity = density_matrix_fidelity_to_pure_target(
                    result_rho, target
                )
                nonzero_indices = set(result_pure.nonzero()[0])
                support_hit = int(nonzero_indices == expected_support)
                support_str = " ".join(
                    format(i, f"0{n}b")[::-1] for i in sorted(nonzero_indices)
                )

                writer.writerow(
                    {
                        "n": n,
                        "run_idx": run_idx,
                        "phase1_shots": args.phase1_shots,
                        "phase2_shots": args.phase2_shots,
                        "fidelity_pure": pure_fidelity,
                        "fidelity_density": density_fidelity,
                        "elapsed_s": elapsed,
                        "support_size": len(nonzero_indices),
                        "support_hit": support_hit,
                        "support_indices": support_str,
                    }
                )

                csv_file.flush()
                os.fsync(csv_file.fileno())

                pure_fidelities.append(pure_fidelity)
                density_fidelities.append(density_fidelity)
                elapsed_times.append(elapsed)
                support_hits += support_hit
                run_idx += 1

                avg_pure_fid = float(np.mean(pure_fidelities))
                avg_density_fid = float(np.mean(density_fidelities))
                avg_elapsed = float(np.mean(elapsed_times))
                print(
                    f"\r{n:>2} | {run_idx + 1:>4}/{args.runs:<4} | "
                    f"avg_fid_pure {avg_pure_fid:>7.6f} | "
                    f"avg_fid_rho {avg_density_fid:>7.6f} | "
                    f"avg_elapsed_s {avg_elapsed:>9.4f}",
                    end="",
                    flush=True,
                )

            print(
                "",
                f"SUMMARY n={n} "
                f"runs={args.runs} "
                f"avg_fid_pure={np.mean(pure_fidelities):.6f} "
                f"std_fid_pure={np.std(pure_fidelities):.6f} "
                f"avg_fid_rho={np.mean(density_fidelities):.6f} "
                f"std_fid_rho={np.std(density_fidelities):.6f} "
                f"avg_elapsed_s={np.mean(elapsed_times):.4f} "
                f"support_hit_rate={support_hits / args.runs:.4f} "
                f"failed_attempts={failed_attempts}",
                flush=True,
            )
    print(f"\nSaved per-run datapoints to: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
