import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from collections import Counter

import src.qutils as qutils


# ----------------------------
# 1. Example: prepare a sparse state
# ----------------------------
def prepare_sparse_state():
    """
    |Psi> = sqrt(0.7)|01> + sqrt(0.3)|10>
    """
    qc = QuantumCircuit(2)
    qc.initialize([0, np.sqrt(0.7), np.sqrt(0.3), 0], [0, 1])
    return qc


# ----------------------------
# 2. Build U_{~Phi} (Section IV)
# ----------------------------
def U_phi_circuit(n, phis):
    qc = QuantumCircuit(n + 1)
    for k in range(1, n + 1):
        qc.rx(-np.pi / 2, 0)
        qc.p(phis[k - 1], 0)
        qc.cx(0, k)
    qc.rx(-np.pi / 2, 0)
    qc.p(phis[n], 0)
    return qc


# ----------------------------
# 3. Full tomography circuit
# ----------------------------
def tomography_circuit(state_prep, phis):
    n = state_prep.num_qubits
    qc = QuantumCircuit(n + 1, n)

    # prepare |Psi>
    qc.append(state_prep.to_gate(), range(1, n + 1))

    # ancilla in |+>
    qc.h(0)

    # Hadamard on data qubits
    qc.h(range(1, n + 1))

    # apply U_phi
    U = U_phi_circuit(n, phis)
    qc.append(U.to_gate(), range(n + 1))

    # inverse Hadamard
    qc.h(range(1, n + 1))

    # measure data qubits
    qc.measure(range(1, n + 1), range(n))
    return qc


def _exp_dir_for_id(exp_root: str, exp_id: int) -> str:
    pattern = os.path.join(exp_root, f"{exp_id:03d}_*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No directory for experiment {exp_id} under {exp_root} (expected {pattern})"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous experiment directory for {exp_id}: {matches}")
    return matches[0]

# ----------------------------
# 4. Run and recover support
# ----------------------------
backend = AerSimulator()
sampler = Sampler(backend)
pm = generate_preset_pass_manager(backend=backend, optimization_level=0)

phis = np.random.uniform(0, 2 * np.pi, 3)
exp_dir = _exp_dir_for_id(exp_root, exp_id)
circ = qutils.load_from_experiment_dir(exp_dir)
circ.barrier()
qc = tomography_circuit(state_prep, phis)

transpiled_circuit = pm.run(qc)

result = sampler.run([transpiled_circuit], shots=4000).result()
counts = result.get_counts()

# Classical recovery
support = Counter(counts)
total = sum(support.values())

recovered_state = {
    basis: count / total for basis, count in support.items() if count / total > 0.05
}

print("Recovered sparse basis states:")
for k, v in recovered_state.items():
    print(f"|{k}>  amplitude^2 ≈ {v:.3f}")

