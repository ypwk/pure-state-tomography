#!/usr/bin/env python3
"""Sparse-measurement pure-state tomography pipeline (paper-style).

This script follows the core workflow in
"Reconstructing Quantum States from Sparse Measurements" (Electronics 2023, 12, 1096):
1) sample random product operators O_j = \\otimes_l m_{i_l} from the paper's exact local basis m0..m3,
2) estimate y_j = <phi|O_j|phi> from shot-based Qiskit measurements (SamplerV2),
3) initialize an MPS via SciPy SVD,
4) optimize MPS tensors by BFGS on Eq. (10)-style MSE loss,
5) return reconstructed pure-state statevector.

Notes:
- The script intentionally uses the exact m0..m3 operators from the paper, including non-Hermitian m1,m2.
- Because Sampler measures Pauli observables, each O_j is expanded into a Pauli-sum exactly.
- For large n and large M, runtime can be substantial.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.linalg import svd
import torch
import yaml
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import src.qutils as qutils


Pauli = str
OpTuple = Tuple[int, ...]


def paper_local_basis() -> List[np.ndarray]:
    """Return paper's local basis matrices [m0,m1,m2,m3]."""
    m0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    m1 = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    m2 = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    m3 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    return [m0, m1, m2, m3]


def op_tuple_to_local_mats(op: OpTuple, basis: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [basis[i] for i in op]


def sample_operator_tuples(n: int, m_count: int, rng: np.random.Generator) -> List[OpTuple]:
    total = 4**n
    if m_count > total:
        raise ValueError(f"Requested M={m_count}, but only {total} unique operators exist for n={n}.")

    # Sample unique operators by integer encoding in base-4.
    picks = rng.choice(total, size=m_count, replace=False)
    ops: List[OpTuple] = []
    for x in picks:
        digits = [0] * n
        v = int(x)
        for idx in range(n - 1, -1, -1):
            digits[idx] = v % 4
            v //= 4
        ops.append(tuple(digits))
    return ops


def local_m_to_pauli_terms(local_idx: int) -> List[Tuple[complex, Pauli]]:
    """Expand one local paper operator m_i into Pauli basis.

    m0 = (I + Z)/2
    m3 = (I - Z)/2
    m1 = (X + iY)/2
    m2 = (X - iY)/2
    """
    if local_idx == 0:
        return [(0.5, "I"), (0.5, "Z")]
    if local_idx == 3:
        return [(0.5, "I"), (-0.5, "Z")]
    if local_idx == 1:
        return [(0.5, "X"), (0.5j, "Y")]
    if local_idx == 2:
        return [(0.5, "X"), (-0.5j, "Y")]
    raise ValueError(f"Invalid local index {local_idx}; expected 0..3")


def expand_op_tuple_to_pauli_sum(op: OpTuple) -> Dict[Pauli, complex]:
    """Expand O=⊗ m_{i_l} into Pauli strings with complex coefficients."""
    acc: Dict[Pauli, complex] = {"": 1.0 + 0.0j}
    for idx in op:
        terms = local_m_to_pauli_terms(idx)
        nxt: Dict[Pauli, complex] = {}
        for p_prefix, c_prefix in acc.items():
            for c_loc, p_loc in terms:
                key = p_prefix + p_loc
                nxt[key] = nxt.get(key, 0.0 + 0.0j) + c_prefix * c_loc
        acc = nxt
    return acc


def pauli_measurement_circuit(base: QuantumCircuit, pauli: Pauli) -> QuantumCircuit:
    """Return measured circuit for given Pauli string in qubit index order q0..q{n-1}."""
    n = base.num_qubits
    qc = base.copy()
    qc.name = f"meas_{pauli}"

    for q, p in enumerate(pauli):
        if p == "X":
            qc.h(q)
        elif p == "Y":
            qc.sdg(q)
            qc.h(q)
        elif p in ("Z", "I"):
            pass
        else:
            raise ValueError(f"Unknown Pauli {p}")

    qc.measure_all()
    return qc


def pauli_expectation_from_counts(pauli: Pauli, counts: Dict[str, int]) -> float:
    """Compute <P> from sampled bitstring counts.

    Qiskit bitstring keys are returned most-significant classical bit first; we reverse
    to align with qubit index order used in `pauli` (q0..q{n-1}).
    """
    shots = sum(counts.values())
    if shots == 0:
        return 0.0

    exp_val = 0.0
    active = [i for i, p in enumerate(pauli) if p != "I"]
    for key, ct in counts.items():
        bits = key[::-1]
        eig = 1
        for q in active:
            if bits[q] == "1":
                eig *= -1
        exp_val += eig * ct

    return exp_val / shots


def estimate_pauli_expectations(
    base_circuit: QuantumCircuit,
    paulis: Sequence[Pauli],
    shots: int,
    opt_level: int,
    batch_size: int,
    seed: int,
) -> Dict[Pauli, float]:
    """Estimate all unique Pauli expectations via shot-based SamplerV2 runs."""
    backend = AerSimulator(seed_simulator=seed)
    sampler = SamplerV2(default_shots=shots, seed=seed)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level)

    out: Dict[Pauli, float] = {}
    for start in range(0, len(paulis), batch_size):
        sub = paulis[start : start + batch_size]
        circuits = [pauli_measurement_circuit(base_circuit, p) for p in sub]
        transpiled = pm.run(circuits)
        result = sampler.run(transpiled).result()

        for i, p in enumerate(sub):
            # measure_all() creates one classical register; use first key robustly.
            data_keys = list(result[i].data.keys())
            if not data_keys:
                raise RuntimeError("Sampler result has no classical data keys.")
            bitarr = result[i].data[data_keys[0]]
            counts = bitarr.get_counts()
            out[p] = pauli_expectation_from_counts(p, counts)

    return out


def mps_expectation(tensors: Sequence[np.ndarray], local_ops: Sequence[np.ndarray]) -> complex:
    """Compute <psi_MPS| O |psi_MPS> by left-to-right transfer contraction."""
    env = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for a, op in zip(tensors, local_ops):
        # env_{kl} = sum_{ij,s,t} env_{ij} A_{i,s,k} op_{s,t} conj(A_{j,t,l})
        env = np.einsum("ij,isk,st,jtl->kl", env, a, op, a.conj(), optimize=True)
    return env[0, 0]


def mps_expectation_torch(tensors: Sequence[torch.Tensor], local_ops: Sequence[torch.Tensor]) -> torch.Tensor:
    """Torch version of <psi_MPS| O |psi_MPS> (complex scalar)."""
    env = torch.ones((1, 1), dtype=torch.complex128, device=tensors[0].device)
    for a, op in zip(tensors, local_ops):
        env = torch.einsum("ij,isk,st,jtl->kl", env, a, op, torch.conj(a))
    return env[0, 0]


def mps_to_statevector(tensors: Sequence[np.ndarray]) -> np.ndarray:
    """Convert MPS tensors [D_l,2,D_r] into a full statevector."""
    work = np.einsum("asb->sb", tensors[0], optimize=True)  # (2, D1)
    for t in tensors[1:]:
        work = np.einsum("ad,dbr->abr", work, t, optimize=True)
        work = work.reshape(work.shape[0] * work.shape[1], work.shape[2])
    if work.shape[1] != 1:
        raise ValueError("Last bond dimension is not 1; invalid open-boundary MPS.")
    vec = work[:, 0]
    nrm = np.linalg.norm(vec)
    if nrm > 0:
        vec = vec / nrm
    return vec


def statevector_to_mps_svd(state: np.ndarray, n: int, bond_dim: int) -> List[np.ndarray]:
    """SVD decomposition into open-boundary MPS with max bond dimension `bond_dim`."""
    psi = state.reshape([2] * n)
    tensors: List[np.ndarray] = []
    left_dim = 1
    work = psi

    for site in range(n - 1):
        work = work.reshape(left_dim * 2, -1)
        u, s, vh = svd(work, full_matrices=False)
        chi = min(bond_dim, s.shape[0])

        u = u[:, :chi]
        s = s[:chi]
        vh = vh[:chi, :]

        a = u.reshape(left_dim, 2, chi)
        tensors.append(a)

        work = np.diag(s) @ vh
        left_dim = chi

    final = work.reshape(left_dim, 2, 1)
    tensors.append(final)
    return tensors


def random_statevector(n: int, rng: np.random.Generator) -> np.ndarray:
    dim = 2**n
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v = v.astype(np.complex128)
    v /= np.linalg.norm(v)
    return v


def flatten_mps_real_imag(tensors: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    shapes = [t.shape for t in tensors]
    parts = []
    for t in tensors:
        parts.append(t.real.ravel())
        parts.append(t.imag.ravel())
    return np.concatenate(parts), shapes


def unflatten_mps_real_imag(x: np.ndarray, shapes: Sequence[Tuple[int, int, int]]) -> List[np.ndarray]:
    tensors: List[np.ndarray] = []
    pos = 0
    for shp in shapes:
        sz = int(np.prod(shp))
        real = x[pos : pos + sz].reshape(shp)
        pos += sz
        imag = x[pos : pos + sz].reshape(shp)
        pos += sz
        tensors.append((real + 1j * imag).astype(np.complex128))
    return tensors


def unflatten_mps_real_imag_torch(
    x: torch.Tensor, shapes: Sequence[Tuple[int, int, int]]
) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    pos = 0
    for shp in shapes:
        sz = int(np.prod(shp))
        real = x[pos : pos + sz].reshape(shp)
        pos += sz
        imag = x[pos : pos + sz].reshape(shp)
        pos += sz
        tensors.append(torch.complex(real, imag))
    return tensors


@dataclass
class TomographyRunResult:
    n: int
    exp_dir: str
    m_count: int
    pauli_count: int
    final_loss: float
    success: bool
    message: str
    fidelity_to_ideal: float
    reconstructed_statevector: np.ndarray
    loss_history: List[float]


def choose_default_m(n: int) -> int:
    # User-approved default: ~25% of full basis, with a practical cap.
    return int(min(round(0.25 * (4**n)), 5000))


def find_exp_dir_for_n(exp_root: str, n: int) -> str:
    candidates = sorted(glob.glob(os.path.join(exp_root, "*_auto")))
    for d in candidates:
        circ = qutils.load_from_experiment_dir(d)
        if circ is not None and circ.num_qubits == n:
            return d
    raise FileNotFoundError(f"No experiment directory under {exp_root} with circuit qubit count n={n}.")


def run_single_tomography(
    exp_dir: str,
    m_count: int,
    shots: int,
    bond_dim: int | None,
    bfgs_maxiter: int,
    opt_level: int,
    batch_size: int,
    seed: int,
    device: str,
) -> TomographyRunResult:
    rng = np.random.default_rng(seed)

    circ = qutils.load_from_experiment_dir(exp_dir)
    if circ is None:
        raise FileNotFoundError(f"{exp_dir}: missing circuit.qasm (or unsupported format).")
    n = circ.num_qubits

    if bond_dim is None:
        # Paper uses D=n for W_n; this default follows that style.
        bond_dim = n

    local_basis = paper_local_basis()
    op_tuples = sample_operator_tuples(n=n, m_count=m_count, rng=rng)

    pauli_sums: List[Dict[Pauli, complex]] = [expand_op_tuple_to_pauli_sum(op) for op in op_tuples]
    unique_paulis = sorted({p for terms in pauli_sums for p in terms.keys()})

    pauli_expectations = estimate_pauli_expectations(
        base_circuit=circ,
        paulis=unique_paulis,
        shots=shots,
        opt_level=opt_level,
        batch_size=batch_size,
        seed=seed,
    )

    y = np.zeros(m_count, dtype=np.complex128)
    for j, terms in enumerate(pauli_sums):
        val = 0.0 + 0.0j
        for p, coeff in terms.items():
            val += coeff * pauli_expectations[p]
        y[j] = val

    # SVD-based random initialization of MPS.
    init_state = random_statevector(n, rng)
    init_tensors = statevector_to_mps_svd(init_state, n=n, bond_dim=bond_dim)

    x0, shapes = flatten_mps_real_imag(init_tensors)
    torch_device = torch.device(device)
    ident = torch.eye(2, dtype=torch.complex128, device=torch_device)

    # Pre-materialize local operator matrices once.
    op_local_mats = [op_tuple_to_local_mats(op, local_basis) for op in op_tuples]
    op_local_mats_torch = [
        [torch.tensor(m, dtype=torch.complex128, device=torch_device) for m in mats]
        for mats in op_local_mats
    ]
    y_torch = torch.tensor(y, dtype=torch.complex128, device=torch_device)

    x = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float64, device=torch_device))
    optimizer = torch.optim.LBFGS(
        [x],
        max_iter=bfgs_maxiter,
        line_search_fn="strong_wolfe",
    )

    loss_history: List[float] = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        tensors = unflatten_mps_real_imag_torch(x, shapes)
        norm = mps_expectation_torch(tensors, [ident] * n)
        # Avoid divide-by-zero; return large penalty without breaking autograd.
        penalty = torch.tensor(1e12, dtype=torch.float64, device=torch_device)
        if torch.abs(norm).item() < 1e-14:
            loss = penalty
            loss_history.append(float(loss.detach().cpu().item()))
            loss.backward()
            return loss

        total = torch.tensor(0.0, dtype=torch.float64, device=torch_device)
        for mats, target in zip(op_local_mats_torch, y_torch):
            pred = mps_expectation_torch(tensors, mats) / norm
            diff = pred - target
            total = total + torch.real(diff.conj() * diff)
        loss = total / len(op_local_mats_torch)
        loss_history.append(float(loss.detach().cpu().item()))
        loss.backward()
        return loss

    opt_res = optimizer.step(closure)

    final_tensors = unflatten_mps_real_imag_torch(x.detach(), shapes)
    final_tensors_np = [t.cpu().numpy() for t in final_tensors]
    final_loss = float(opt_res.detach().cpu().item())
    rec_state = mps_to_statevector(final_tensors_np)

    ideal = qutils.circuit_to_statevector(circ)
    fidelity = float(abs(np.vdot(ideal, rec_state)))

    return TomographyRunResult(
        n=n,
        exp_dir=exp_dir,
        m_count=m_count,
        pauli_count=len(unique_paulis),
        final_loss=final_loss,
        success=True,
        message="LBFGS completed",
        fidelity_to_ideal=fidelity,
        reconstructed_statevector=rec_state,
        loss_history=loss_history,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-style sparse-measurement pure-state tomography")
    p.add_argument("--exp-dir", type=str, default=None, help="Single experiment dir containing circuit.qasm")
    p.add_argument("--exp-root", type=str, default="experiments/configs", help="Root to auto-find exp dirs by n")
    p.add_argument("--n-min", type=int, default=4, help="Default qubit range lower bound if --exp-dir not set")
    p.add_argument("--n-max", type=int, default=10, help="Default qubit range upper bound if --exp-dir not set")

    p.add_argument("--M", type=int, default=None, help="Number of sampled paper operators; default=min(0.25*4^n,5000)")
    p.add_argument("--shots", type=int, default=16384, help="Shots per Pauli measurement circuit")
    p.add_argument("--bond-dim", type=int, default=None, help="MPS bond dimension D; default D=n")
    p.add_argument("--bfgs-maxiter", type=int, default=80, help="BFGS max iterations")
    p.add_argument("--opt-level", type=int, default=2, choices=[0, 1, 2, 3], help="Transpiler optimization level")
    p.add_argument("--batch-size", type=int, default=128, help="Sampler batch size for Pauli circuits")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto|cpu|cuda|cuda:0 etc. (auto picks CUDA when available)",
    )

    p.add_argument("--out", type=str, default="paper_sparse_qst_results.json", help="JSON summary output path")
    p.add_argument(
        "--save-statevectors-dir",
        type=str,
        default="paper_sparse_qst_statevectors",
        help="Directory to save reconstructed statevectors as .npy files",
    )
    p.add_argument(
        "--save-loss-dir",
        type=str,
        default="paper_sparse_qst_losses",
        help="Directory to save loss history as .npy files",
    )
    p.add_argument(
        "--show-loss",
        action="store_true",
        help="Show matplotlib loss curve windows during the run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")

    batch_path = os.path.join("experiments", "batches", "ghz_chain.yaml")
    with open(batch_path, "r", encoding="utf-8") as f:
        batch_cfg = yaml.safe_load(f) or {}
    exp_ids = batch_cfg.get("experiment_ids", [])
    if not exp_ids:
        raise ValueError(f"{batch_path}: missing or empty experiment_ids")
    exp_dirs = [os.path.join(args.exp_root, f"{int(exp_id):03d}_auto") for exp_id in exp_ids]

    os.makedirs(args.save_statevectors_dir, exist_ok=True)
    os.makedirs(args.save_loss_dir, exist_ok=True)

    summaries = []
    for exp_dir in exp_dirs:
        circ = qutils.load_from_experiment_dir(exp_dir)
        if circ is None:
            raise FileNotFoundError(f"{exp_dir}: missing circuit.qasm")
        n = circ.num_qubits
        m_count = args.M if args.M is not None else choose_default_m(n)

        print(f"\n=== Running sparse tomography for n={n}, exp_dir={exp_dir}")
        print(f"M={m_count}, shots={args.shots}, bond_dim={args.bond_dim or n}, bfgs_maxiter={args.bfgs_maxiter}")
        print(f"Using torch device: {device}")

        run = run_single_tomography(
            exp_dir=exp_dir,
            m_count=m_count,
            shots=args.shots,
            bond_dim=args.bond_dim,
            bfgs_maxiter=args.bfgs_maxiter,
            opt_level=args.opt_level,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )

        vec_path = os.path.join(args.save_statevectors_dir, f"reconstructed_n{run.n}.npy")
        np.save(vec_path, run.reconstructed_statevector)
        loss_path = os.path.join(args.save_loss_dir, f"loss_history_n{run.n}.npy")
        np.save(loss_path, np.asarray(run.loss_history, dtype=np.float64))
        if args.show_loss:
            fig, ax = plt.subplots()
            ax.plot(run.loss_history, color="black")
            ax.set_yscale("log")
            ax.set_xlabel("LBFGS Closure Call")
            ax.set_ylabel("Loss")
            ax.set_title(f"Loss Curve (n={run.n})")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()

        summary = {
            "n": run.n,
            "exp_dir": run.exp_dir,
            "M": run.m_count,
            "unique_pauli_count": run.pauli_count,
            "final_loss": run.final_loss,
            "optimizer_success": run.success,
            "optimizer_message": run.message,
            "fidelity_to_ideal": run.fidelity_to_ideal,
            "reconstructed_statevector_path": vec_path,
            "loss_history_path": loss_path,
            "loss_history_len": len(run.loss_history),
        }
        summaries.append(summary)

        print(
            f"Finished n={run.n}: loss={run.final_loss:.6e}, fidelity={run.fidelity_to_ideal:.6f}, "
            f"unique_paulis={run.pauli_count}, success={run.success}"
        )

        if args.show_loss:
            plt.show()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nSaved run summary to {args.out}")


if __name__ == "__main__":
    main()
