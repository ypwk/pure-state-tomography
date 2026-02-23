from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseEstimation
from qiskit_aer import AerSimulator

import src.qutils as qutils
from src.noise_modeling import make_custom_noise_model


def _build_u_phi_gate(num_data_qubits: int, phis: np.ndarray):
    """
    Build the Section-IV U_phi over (ancilla + data) qubits.
    Qubit order in this gate is [ancilla, data_0, ..., data_{n-1}].
    """
    n = num_data_qubits

    if len(phis) != n + 1:
        raise ValueError(
            f"phis must have length {n + 1}, got {len(phis)}"
        )

    # Qubit ordering:
    # q[0] ........ q[n-1] : data (WPD) qubits
    # q[n]                   photon qubit
    qc = QuantumCircuit(n + 1)

    photon = n  # index of photon qubit

    for k in range(n + 1):
        # RX(-pi/2) on photon (beam splitter)
        qc.rx(-np.pi / 2, photon)

        # Phase shift Phi_k on photon
        qc.p(phis[k], photon)

        # CNOT for WPD interaction (only for first n layers)
        if k < n:
            qc.cx(photon, k)

    return qc.to_gate(label="U_Phi")


def build_phase1_circuit(
    state_prep_circuit,
    *,
    phase_register_bits: int,
    phis: np.ndarray | None = None,
):
    def _inverse_qft(qc, qubits):
        """
        In-place inverse QFT on given qubits.
        """
        n = len(qubits)

        # Swap qubits
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - i - 1])

        # Controlled rotations
        for j in range(n):
            for k in range(j):
                qc.cp(-np.pi / (2 ** (j - k)), qubits[j], qubits[k])
            qc.h(qubits[j])

    n = state_prep_circuit.num_qubits
    t = phase_register_bits

    # Registers:
    # [ phase | data | anc ]
    qc = QuantumCircuit(t + n + 1)

    phase = list(range(t))
    data = list(range(t, t + n))
    anc = t + n
    system = data + [anc]

    # 1. Prepare |Psi>
    qc.compose(state_prep_circuit, qubits=data, inplace=True)

    # 2. Ancilla in |+>
    qc.h(anc)

    # 3. H^{⊗n} on data qubits
    for q in data:
        qc.h(q)

    # 4. Initialize phase register
    for q in phase:
        qc.h(q)

    # 5. Controlled-U^{2^k}
    #
    # Phase qubit 0 controls U^{2^{t-1}}
    # Phase qubit t-1 controls U^{2^0}
    #
    for j in range(t):
        repetitions = 2 ** (t - j - 1)

        controlled_U = _build_u_phi_gate(num_data_qubits=n, phis=phis).control()

        for _ in range(repetitions):
            qc.append(
                controlled_U,
                [phase[j]] + system
            )

    # 6. Inverse QFT on phase register
    _inverse_qft(qc, phase)

    # 7. Undo Hadamards on data qubits
    for q in data:
        qc.h(q)

    qc.measure_all()

    return qc


def phase1_recover_support(
    state_prep_circuit,
    aer_sim,
    *,
    phase_register_bits: int,
    shots: int,
    seed: int | None = None,
):
    """
    Recover computational-basis support by marginalizing data-register outcomes.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")

    num_data_qubits = state_prep_circuit.num_qubits

    # rng = np.random.default_rng(seed)
    # phis = rng.uniform(0.0, 2.0 * np.pi, size=num_data_qubits + 1)
    phis = np.zeros(num_data_qubits + 1)
    qc = build_phase1_circuit(
        state_prep_circuit,
        phase_register_bits=phase_register_bits,
        phis=phis,
    )

    probs = qutils.run_circuit(
        aer_sim,
        qc,
        shots=shots,
        batch_size=1,
        base_seed=seed,
    )[0]

    data_dim = 1 << num_data_qubits
    data_probs = np.zeros(data_dim, dtype=float)
    t = phase_register_bits
    n = num_data_qubits
    data_mask = (1 << n) - 1
    for idx, p in enumerate(probs):
        data_index = (idx >> t) & data_mask
        data_probs[data_index] += p

    support = np.flatnonzero(data_probs >= 0.1).astype(int).tolist()
    # support = np.array([0, len(data_probs) - 1])
    # print(f"phase 1 support: {sorted(support)}")
    return sorted(support)
