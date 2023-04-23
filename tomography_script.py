# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit_aer import AerSimulator

# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options

# Loading your IBM Quantum account(s)
service = QiskitRuntimeService(channel="ibm_quantum")

# Invoke a primitive inside a session. For more details see https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials.html
# with Session(backend=service.backend("ibmq_qasm_simulator")):
#     result = Sampler().run(circuits).result()

import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram

def create_circuit(state):
    qc = QuantumCircuit(2)
    qc.initialize(state, [0, 1])
    return qc

def measure_circuit(qc):
    qc.measure_all()
    return qc

def run_circuit(qc):
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc, aer_sim)
    qobj = assemble(t_qc, shots=1024)
    result = aer_sim.run(t_qc, shots=1024).result()
    counts = result.get_counts(qc)
    return counts

def find_nonzero_positions(counts):
    positions = []
    for key in counts:
        if counts[key] > 0:
            positions.append([int(key[0]), int(key[1])])
    return positions

def apply_gates(qc, positions):
    i1, i0 = positions[0]
    j1, j0 = positions[1] if len(positions) > 1 else (None, None)

    # Define gates
    ih = QuantumCircuit(2)
    ih.h(1)

    ihd = QuantumCircuit(2)
    ihd.u(0, -np.pi/4, -np.pi/4, 1)

    hi = QuantumCircuit(2)
    hi.h(0)

    hdi = QuantumCircuit(2)
    hdi.u(0, -np.pi/4, -np.pi/4, 0)

    hic = QuantumCircuit(2)
    hic.h(0)
    hic.cx(0, 1)

    hdi_c = QuantumCircuit(2)
    hdi_c.u(0, -np.pi/4, -np.pi/4, 0)
    hdi_c.cx(0, 1)

    # Apply gates
    if len(positions) == 1:
        pass
    elif i1 == j1:
        qc.append(ih, [0, 1])
        qc.append(ihd, [0, 1])
    elif i0 == j0:
        qc.append(hi, [0, 1])
        qc.append(hdi, [0, 1])
    else:
        qc.append(hic, [0, 1])
        qc.append(hdi_c, [0, 1])

    return qc

def main():
    # Define initial states
    initial_states = [
        (np.array([1/2, 1/np.sqrt(2), 1/np.sqrt(6), 1/np.sqrt(12)]), "col 1"),
        (np.array([1/2, -1/np.sqrt(2), 1/np.sqrt(6), 1/np.sqrt(12)]), "col 2"),
        (np.array([1/2, 0, -2/np.sqrt(6), 1/np.sqrt(12)]), "col 3"),
        (np.array([1/2, 0, 0, -3/np.sqrt(12)]), "col 4"),
    ]

    all_results = {}
    for state, state_name in initial_states:
        qc = create_circuit(state)

        # Measure |ψ⟩ to get information |x1|^2, ..., |x4|^2
        qc_measure = measure_circuit(qc.copy())
        counts = run_circuit(qc_measure)

        # Find positions with nonzero entries
        positions = find_nonzero_positions(counts)

        # Apply gates based on conditions
        qc_gates = apply_gates(qc.copy(), positions)

        # Measure circuit with applied gates
        qc_gates_measure = measure_circuit(qc_gates.copy())
        counts_gates = run_circuit(qc_gates_measure)

        # Store results
        all_results[state_name] = {
            'initial_counts': counts,
            'final_counts': counts_gates
        }

    # Print and visualize results
    for state_name, results in all_results.items():
        print(f"Results for state {state_name}:")
        print("Initial measurements:")
        print(results['initial_counts'])
        plot_histogram(results['initial_counts']).show()

        print("Measurements after applying gates:")
        print(results['final_counts'])
        plot_histogram(results['final_counts']).show()
        print()

if __name__ == "__main__":
    main()
