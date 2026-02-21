#!/usr/bin/env python
"""K-sparse pure-state tomography with phase estimation (small qubit counts).

This module implements a practical two-phase workflow inspired by
Gulbahar (2021), "K-sparse Pure State Tomography with Phase Estimation":

1) Discover support basis states with phase estimation over the Section IV
   circuit construction U_phi.
2) Recover complex coefficients on that support by linear least squares over
   measurement probabilities from local Pauli-basis settings.

The implementation targets small systems (n < 10) and ideal simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import src.qutils as qutils


@dataclass
class ReconstructionResult:
    statevector: np.ndarray
    support_indices: list[int]
    support_probabilities: dict[int, float]
    phase_register_bits: int


def build_u_phi_circuit(
    num_data_qubits: int, phases: Sequence[float] | None = None
):
    """Build U_phi from paper Section IV on (num_data_qubits + 1) qubits.

    Qubit layout:
    - q[0 : num_data_qubits] are data/WPD qubits
    - q[num_data_qubits] is the ancillary path/photon qubit
    """
    _require_qiskit()
    from qiskit import QuantumCircuit

    if num_data_qubits <= 0:
        raise ValueError("num_data_qubits must be positive")

    if phases is None:
        phases = _default_phases(num_data_qubits + 1)
    if len(phases) != num_data_qubits + 1:
        raise ValueError("phases must have length num_data_qubits + 1")

    n = num_data_qubits
    anc = n
    qc = QuantumCircuit(n + 1, name="U_phi")

    # n stages with interaction to each WPD/data qubit
    for k in range(n):
        qc.rx(-np.pi / 2, anc)
        qc.p(float(phases[k]), anc)
        qc.cx(anc, k)

    # final stage (no WPD interaction)
    qc.rx(-np.pi / 2, anc)
    qc.p(float(phases[n]), anc)
    return qc


def reconstruct_k_sparse_state(
    state_prep_circuit,
    *,
    phases: Sequence[float] | None = None,
    phase_register_bits: int | None = None,
    phase1_shots: int = 4096,
    phase2_shots: int = 8192,
    support_probability_threshold: float | None = None,
    backend=None,
    seed: int | None = None,
) -> ReconstructionResult:
    """Run both phases and return reconstructed full statevector."""
    _require_qiskit()
    n = state_prep_circuit.num_qubits
    if n <= 0:
        raise ValueError("Input circuit must have at least one qubit")
    if n >= 10:
        raise ValueError("This implementation targets n < 10")

    if phase_register_bits is None:
        phase_register_bits = max(4, n + 1)

    if backend is None:
        backend = build_backend()

    support_indices, support_probs = estimate_support_with_phase_estimation(
        state_prep_circuit,
        phases=phases,
        phase_register_bits=phase_register_bits,
        shots=phase1_shots,
        support_probability_threshold=support_probability_threshold,
        backend=backend,
        seed=seed,
    )

    reconstructed = recover_coefficients_least_squares(
        state_prep_circuit,
        support_indices=support_indices,
        shots=phase2_shots,
        backend=backend,
        seed=None if seed is None else seed + 1_000_003,
    )

    return ReconstructionResult(
        statevector=reconstructed,
        support_indices=support_indices,
        support_probabilities=support_probs,
        phase_register_bits=phase_register_bits,
    )


def estimate_support_with_phase_estimation(
    state_prep_circuit,
    *,
    phases: Sequence[float] | None = None,
    phase_register_bits: int = 6,
    shots: int = 4096,
    support_probability_threshold: float | None = None,
    backend=None,
    seed: int | None = None,
) -> tuple[list[int], dict[int, float]]:
    """Estimate support basis indices (phase 1)."""
    _require_qiskit()
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import QFT

    n = state_prep_circuit.num_qubits
    if n <= 0:
        raise ValueError("Input circuit must have at least one qubit")

    phase = QuantumRegister(phase_register_bits, "phase")
    data = QuantumRegister(n, "data")
    anc = QuantumRegister(1, "anc")
    c_data = ClassicalRegister(n, "c_data")
    qc = QuantumCircuit(phase, data, anc, c_data, name="phase1_support")

    # Step 2 (paper): prepare |Psi> on data, plus ancilla, and H^n on data.
    qc.compose(state_prep_circuit, qubits=list(data), inplace=True)
    qc.h(anc[0])
    qc.h(data)

    # Standard QPE register preparation.
    qc.h(phase)

    if phases is None:
        phases = _default_phases(n + 1)
    if len(phases) != n + 1:
        raise ValueError("phases must have length num_data_qubits + 1")

    data_qubits = list(data)
    for j in range(phase_register_bits):
        repeats = 2**j
        for _ in range(repeats):
            _append_controlled_u_phi_once(
                qc,
                ctrl_qubit=phase[j],
                data_qubits=data_qubits,
                anc_qubit=anc[0],
                phases=phases,
            )

    iqft = QFT(
        num_qubits=phase_register_bits, inverse=True, do_swaps=False, name="iqft"
    ).decompose()
    qc.compose(iqft, qubits=list(phase), inplace=True)

    # Step 4/5 (paper): apply H^n back and measure data qubits.
    qc.h(data)
    qc.measure(data, c_data)

    probs = _sample_probability_dict(
        qc,
        num_measured_bits=n,
        shots=shots,
        backend=backend,
        seed=seed,
    )

    if support_probability_threshold is None:
        support_probability_threshold = max(1.0 / shots, 0.005)

    support = [
        idx
        for idx, p in sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
        if p >= support_probability_threshold
    ]
    if not support and probs:
        support = [max(probs, key=probs.get)]

    support_probabilities = {idx: probs[idx] for idx in support}
    return support, support_probabilities


def recover_coefficients_least_squares(
    state_prep_circuit,
    *,
    support_indices: Sequence[int],
    shots: int = 8192,
    basis_settings: Iterable[str] | None = None,
    backend=None,
    seed: int | None = None,
) -> np.ndarray:
    """Recover full statevector amplitudes using support-projected LS (phase 2)."""
    _require_qiskit()
    from qiskit import QuantumCircuit

    n = state_prep_circuit.num_qubits
    dim = 2**n
    support = sorted(set(int(s) for s in support_indices))
    if not support:
        raise ValueError("support_indices must be non-empty")
    if any(s < 0 or s >= dim for s in support):
        raise ValueError("support_indices contains invalid basis index")

    if basis_settings is None:
        basis_list = _default_basis_settings(num_qubits=n, support_dim=len(support))
    else:
        basis_list = list(basis_settings)

    rows: list[np.ndarray] = []
    y_vals: list[float] = []
    meas_circuits = []

    for basis in basis_list:
        if len(basis) != n or any(b not in "XYZ" for b in basis):
            raise ValueError("Each basis setting must be a length-n string in {X,Y,Z}")

        meas = QuantumCircuit(n, n, name=f"meas_{basis}")
        meas.compose(state_prep_circuit, qubits=list(range(n)), inplace=True)
        _apply_measurement_basis_rotation(meas, basis)
        meas.measure(range(n), range(n))
        meas_circuits.append(meas)

    prob_dicts = _sample_probability_dicts(
        meas_circuits,
        num_measured_bits=n,
        shots=shots,
        backend=backend,
        seed=seed,
    )

    for basis, prob_by_outcome in zip(basis_list, prob_dicts):
        for outcome in range(2**n):
            p = prob_by_outcome.get(outcome, 0.0)
            m = _restricted_measurement_operator_row(
                basis=basis,
                outcome_index=outcome,
                support_indices=support,
                num_qubits=n,
            )
            rows.append(m.reshape(-1))
            y_vals.append(float(p))

    a_complex = np.asarray(rows, dtype=np.complex128)
    y = np.asarray(y_vals, dtype=np.float64)
    rho = _solve_density_matrix_least_squares(a_complex, y, support_dim=len(support))

    eigvals, eigvecs = np.linalg.eigh(rho)
    principal = eigvecs[:, int(np.argmax(eigvals))]

    # Fix global phase by making first non-negligible component real-positive.
    nz = np.where(np.abs(principal) > 1e-12)[0]
    if nz.size > 0:
        principal = principal * np.exp(-1j * np.angle(principal[nz[0]]))

    full = np.zeros(dim, dtype=np.complex128)
    for amp, idx in zip(principal, support):
        full[idx] = amp

    norm = np.linalg.norm(full)
    if norm > 0:
        full /= norm
    return full


def _restricted_measurement_operator_row(
    *,
    basis: str,
    outcome_index: int,
    support_indices: Sequence[int],
    num_qubits: int,
) -> np.ndarray:
    """Return restricted projector row M for one (basis, outcome)."""
    k = len(support_indices)
    out_bits = _int_to_msb_bits(outcome_index, num_qubits)
    support_bits = [_int_to_msb_bits(s, num_qubits) for s in support_indices]

    v = np.zeros(k, dtype=np.complex128)
    for a, bits_s in enumerate(support_bits):
        amp = 1.0 + 0.0j
        for q in range(num_qubits):
            amp *= _single_qubit_overlap(basis[q], out_bits[q], bits_s[q])
        v[a] = amp
    return np.outer(np.conjugate(v), v)


def _single_qubit_overlap(axis: str, out_bit: int, state_bit: int) -> complex:
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    if axis == "Z":
        return 1.0 if out_bit == state_bit else 0.0
    if axis == "X":
        if out_bit == 0:
            return inv_sqrt2
        return inv_sqrt2 if state_bit == 0 else -inv_sqrt2
    if axis == "Y":
        # U = H * Sdg for "sdg; h; measure-z"
        if out_bit == 0:
            return inv_sqrt2 if state_bit == 0 else -1j * inv_sqrt2
        return inv_sqrt2 if state_bit == 0 else 1j * inv_sqrt2
    raise ValueError(f"Unsupported axis: {axis}")


def _apply_measurement_basis_rotation(circuit, basis: str) -> None:
    for q, axis in enumerate(basis):
        if axis == "X":
            circuit.h(q)
        elif axis == "Y":
            circuit.sdg(q)
            circuit.h(q)
        elif axis == "Z":
            pass
        else:
            raise ValueError(f"Unsupported basis axis: {axis}")


def _append_controlled_u_phi_once(
    circuit, *, ctrl_qubit, data_qubits: Sequence, anc_qubit, phases: Sequence[float]
) -> None:
    """Append one controlled-U_phi application using primitive controlled gates."""
    n = len(data_qubits)
    for k in range(n):
        circuit.crx(-np.pi / 2, ctrl_qubit, anc_qubit)
        circuit.cp(float(phases[k]), ctrl_qubit, anc_qubit)
        circuit.ccx(ctrl_qubit, anc_qubit, data_qubits[k])
    circuit.crx(-np.pi / 2, ctrl_qubit, anc_qubit)
    circuit.cp(float(phases[n]), ctrl_qubit, anc_qubit)


def _solve_density_matrix_least_squares(
    a_complex: np.ndarray, y: np.ndarray, support_dim: int
) -> np.ndarray:
    """Solve for rho in y = Tr(M rho), then project to physical density matrix."""
    a_re = a_complex.real
    a_im = a_complex.imag
    lhs = np.block([[a_re, -a_im], [a_im, a_re]])
    rhs = np.concatenate([y, np.zeros_like(y)])

    z, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    n = support_dim * support_dim
    x = z[:n] + 1j * z[n:]
    rho = x.reshape((support_dim, support_dim))

    # Physical projection: Hermitian, PSD, trace-1.
    rho = 0.5 * (rho + rho.conj().T)
    w, v = np.linalg.eigh(rho)
    w = np.clip(w.real, 0.0, None)
    if np.sum(w) <= 0:
        w[:] = 0.0
        w[np.argmax(np.abs(np.diag(rho)))] = 1.0
    else:
        w = w / np.sum(w)
    return (v * w) @ v.conj().T


def _sample_probability_dict(
    circuit, *, num_measured_bits: int, shots: int, backend=None, seed: int | None = None
) -> dict[int, float]:
    return _sample_probability_dicts(
        [circuit],
        num_measured_bits=num_measured_bits,
        shots=shots,
        backend=backend,
        seed=seed,
    )[0]


def _sample_probability_dicts(
    circuits, *, num_measured_bits: int, shots: int, backend=None, seed: int | None = None
) -> list[dict[int, float]]:
    if backend is None:
        backend = build_backend()

    max_outcomes = 1 << num_measured_bits
    all_probs: list[dict[int, float]] = []

    for idx, circuit in enumerate(circuits):
        circuit_seed = None if seed is None else int(seed + idx)
        probs = qutils.run_circuit(
            backend,
            circuit,
            shots=shots,
            batch_size=1,
            base_seed=circuit_seed,
        )[0]
        all_probs.append(
            {
                idx: float(p)
                for idx, p in enumerate(probs[:max_outcomes])
                if float(p) > 0.0
            }
        )

    return all_probs


def _extract_probabilities_from_sampler_result(
    *, result, num_measured_bits: int
) -> list[dict[str, float]]:
    """Best-effort parser across SamplerV2 result variants."""
    # Variant 1: legacy-like quasi distribution list
    quasi_dists = getattr(result, "quasi_dists", None)
    if quasi_dists is not None and len(quasi_dists) > 0:
        all_probs = []
        for quasi in quasi_dists:
            all_probs.append(
                _normalize_probability_map(
                    {
                        _format_outcome_key(k, num_measured_bits): float(v)
                        for k, v in quasi.items()
                    }
                )
            )
        return all_probs

    # Variant 2: PubResult style (result[0].data.<creg>.get_counts())
    try:
        all_probs = []
        for pub in result:
            data = getattr(pub, "data", None)
            if data is None:
                continue

            meas = getattr(data, "meas", None)
            if meas is not None and hasattr(meas, "get_counts"):
                all_probs.append(
                    _counts_to_probabilities(
                        meas.get_counts(), num_measured_bits=num_measured_bits
                    )
                )
                continue

            found = None
            for name in dir(data):
                if name.startswith("_"):
                    continue
                obj = getattr(data, name)
                if hasattr(obj, "get_counts"):
                    found = _counts_to_probabilities(
                        obj.get_counts(), num_measured_bits=num_measured_bits
                    )
                    break
            if found is not None:
                all_probs.append(found)

        if all_probs:
            return all_probs
    except Exception:
        pass

    raise RuntimeError("Unsupported sampler result format; could not extract probabilities")


def _counts_to_probabilities(counts: dict, *, num_measured_bits: int) -> dict[str, float]:
    cleaned = {}
    total = 0.0
    for raw_key, value in counts.items():
        bits = _format_outcome_key(raw_key, num_measured_bits)
        v = float(value)
        cleaned[bits] = cleaned.get(bits, 0.0) + v
        total += v
    if total <= 0:
        return {}
    return {k: v / total for k, v in cleaned.items()}


def _normalize_probability_map(prob: dict[str, float]) -> dict[str, float]:
    total = float(sum(prob.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in prob.items()}


def _format_outcome_key(key, num_measured_bits: int) -> str:
    if isinstance(key, int):
        return format(key, f"0{num_measured_bits}b")
    s = str(key).replace(" ", "")
    if all(ch in "01" for ch in s):
        if len(s) < num_measured_bits:
            s = s.zfill(num_measured_bits)
        elif len(s) > num_measured_bits:
            s = s[-num_measured_bits:]
        return s
    raise ValueError(f"Cannot parse outcome key: {key}")


def _default_phases(length: int) -> list[float]:
    # Uniform phase spacing in [0, 2pi) is used in the paper's numerical section.
    return [2.0 * np.pi * j / length for j in range(length)]


def _default_basis_settings(num_qubits: int, support_dim: int) -> list[str]:
    """Compact Pauli-basis set for practical sparse recovery.

    Uses O(support_dim log support_dim) random settings, independent from 3^n.
    """
    base = {
        "Z" * num_qubits,
        "X" * num_qubits,
        "Y" * num_qubits,
    }
    for q in range(num_qubits):
        x_local = ["Z"] * num_qubits
        y_local = ["Z"] * num_qubits
        x_local[q] = "X"
        y_local[q] = "Y"
        base.add("".join(x_local))
        base.add("".join(y_local))

    target = max(len(base), int(np.ceil(8 * support_dim * np.log2(max(2, support_dim)))))
    target = min(target, 64, 3**num_qubits)

    rng = np.random.default_rng(7)
    axes = np.array(list("XYZ"))
    settings = set(base)
    while len(settings) < target:
        s = "".join(rng.choice(axes, size=num_qubits, replace=True).tolist())
        settings.add(s)
    return sorted(settings)


def _int_to_msb_bits(value: int, num_bits: int) -> list[int]:
    return [int(ch) for ch in format(value, f"0{num_bits}b")]


def _build_sampler(*, backend):
    _require_qiskit()
    # Prefer Aer native primitive for Aer backend.
    if "AerSimulator" in backend.__class__.__name__:
        from qiskit_aer.primitives import SamplerV2

        for kwargs in ({"backend": backend}, {"mode": backend}, {}):
            try:
                return SamplerV2(**kwargs)
            except TypeError:
                continue
        return SamplerV2()

    # Prefer backend-bound primitive when available.
    try:
        from qiskit.primitives import BackendSamplerV2

        return BackendSamplerV2(backend=backend)
    except Exception:
        pass

    # Generic fallback.
    from qiskit_aer.primitives import SamplerV2

    for kwargs in ({"backend": backend}, {"mode": backend}, {}):
        try:
            return SamplerV2(**kwargs)
        except TypeError:
            continue
    return SamplerV2()


def build_backend():
    _require_qiskit()
    from qiskit_aer import AerSimulator
    from src.noise_modeling import make_custom_noise_model

    noise_model = make_custom_noise_model(
        p_1q=4.239e-4,
        p_2q=3.416e-3,
        p_meas=1.0e-2,
        coherent_phase=0.0,
        n_qubits=32,
    )
    return AerSimulator(noise_model=noise_model, method="density_matrix", device="CPU")


def _build_backend():
    return build_backend()


def _transpile_for_backend(circuit, backend):
    from qiskit import transpile

    return transpile(circuit, backend=backend, optimization_level=0)


def _backend_requires_transpile(backend) -> bool:
    return "AerSimulator" not in backend.__class__.__name__


def _require_qiskit() -> None:
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "This module requires qiskit and qiskit-aer. "
            "Install with: pip install qiskit qiskit-aer"
        ) from exc
