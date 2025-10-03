from numpy import ndarray, sqrt, asarray, zeros
import numpy as np
from datetime import datetime as dt
import os

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

import src.putils as putils
import src.qutils as qutils

MAX_CONC_JOB_COUNT = 3


class measurement_manager:
    def __init__(self, n_shots, execution_type, verbose: bool, partial_mixing: bool = False) -> None:
        self.n_shots = n_shots
        self.execution_type = execution_type
        self.verbose = verbose
        self.m_state = None
        self.clean_m_state = None
        self.partial_mixing = partial_mixing

        self.num_measurements = 0

        # Sacred-compatible logging: just print if verbose=True
        self.verboseprint = print if verbose else (lambda *a, **k: None)

        self.__measurements = None
        self.__c_measurements = None
        self.__clean_measurements = None

        if self.execution_type == qutils.execution_type.simulator:
            self.aer_sim = AerSimulator(
                method="matrix_product_state", provider=FakeBrisbane(), device="GPU"
            )
            self.verboseprint("Available devices:",
                              self.aer_sim.available_devices())

    # ---------- State setup ----------
    def set_state(self, tomography_type, state: ndarray | QuantumCircuit) -> None:
        """Sets the state for tomography (state or process)."""
        if isinstance(state, ndarray):
            if tomography_type == qutils.tomography_type.state:
                self.verboseprint(f"Input vector: {state}")
                self.n_qubits = putils.fast_log2(len(state))
                self.m_state = qutils.create_vector_circuit(
                    state, self.n_qubits)
            elif tomography_type == qutils.tomography_type.process:
                if state.shape[0] != state.shape[1]:
                    raise ValueError("Process matrix must be square.")
                self.verboseprint(f"Input matrix:\n{state}")
                self.n_qubits = putils.fast_log2(state.shape[0]) * 2
                self.m_state = qutils.create_matrix_circuit(
                    state, self.n_qubits)
        elif isinstance(state, QuantumCircuit):
            if tomography_type == qutils.tomography_type.state:
                self.verboseprint(f"Input circuit:\n{state}")
                self.n_qubits = state.num_qubits
                self.m_state = state
                self.verboseprint(
                    f"Statevector:\n{qutils.circuit_to_statevector(self.m_state)}"
                )
            else:
                self.n_qubits = state.num_qubits * 2
                self.m_state = QuantumCircuit(self.n_qubits)
                for a in range(self.n_qubits // 2):
                    self.m_state.h(self.n_qubits // 2 + a)
                for a in range(self.n_qubits // 2 - 1, -1, -1):
                    self.m_state.cx(self.n_qubits // 2 + a, a)
                self.m_state = self.m_state.compose(
                    state.copy(), range(0, state.num_qubits)
                )

        self.m_state.barrier()
        self.clean_m_state = self.m_state.copy("clean")

        # reset measurement stores
        self.__measurements = {t: [None] *
                               self.n_qubits for t in qutils.m_type}
        self.__c_measurements = {t: [] for t in qutils.m_type}
        self.__clean_measurements = {
            t: [None] * self.n_qubits for t in qutils.m_type}
        self.num_measurements = 0

    # ---------- Circuit builder ----------
    def build_circuit(self, measure_type, op_pos: int, cnots=(), clean=False):
        """Construct circuit for a given measurement type and operator position."""
        base = self.clean_m_state if clean else self.m_state
        qc = base.copy("execute")
        qc.barrier()

        # Insert CNOTs or Hadamards
        if self.partial_mixing:
            qubit_set = set()
            for c in cnots or []:
                qubit_set.add(c[0])
                qubit_set.add(c[1])
            for loc in qubit_set:
                qc.h(loc)
        else:
            for c in cnots or []:
                ctrl, tgt = self.n_qubits - c[0] - 1, self.n_qubits - c[1] - 1
                qc.cx(ctrl, tgt)
            if cnots:
                qc.barrier()

        # Apply operator
        q = self.n_qubits - op_pos - 1
        if measure_type == qutils.m_type.real_hadamard:
            qc.h(q)
        elif measure_type == qutils.m_type.cmplx_hadamard:
            qc.unitary([[1/sqrt(2), 1j/sqrt(2)], [1/sqrt(2), -1j/sqrt(2)]], q)
        elif measure_type == qutils.m_type.identity:
            qc.id(q)
        return qc

    # ---------- Measurements ----------
    def add_measurement(self, measure_type, op_pos=0, cnots=(), clean=False):
        """Perform measurement and store results in appropriate registry."""
        qc = self.build_circuit(measure_type, op_pos, cnots, clean)
        res = self.measure_state(qc)
        self.num_measurements += 1
        entry = {"res": res, "str": str(qc) if self.verbose else "Not Verbose"}

        if cnots:
            self.__c_measurements[measure_type].append(
                {"cnots": cnots, "op_pos": op_pos,
                    "data": res, "str": entry["str"]}
            )
        elif clean:
            self.__clean_measurements[measure_type][op_pos] = entry
        else:
            self.__measurements[measure_type][op_pos] = entry
        return res

    def fetch(self, measure_type, op_pos=0, cnots=(), clean=False):
        """Retrieve measurement, adding it if not already present."""
        if cnots:
            for e in self.__c_measurements[measure_type]:
                if e["cnots"] == cnots and e["op_pos"] == op_pos:
                    self.verboseprint(e["str"])
                    return e["data"]
            return self.add_measurement(measure_type, op_pos, cnots)
        store = self.__clean_measurements if clean else self.__measurements
        if store[measure_type][op_pos] is None:
            return self.add_measurement(measure_type, op_pos, clean=clean)
        self.verboseprint(store[measure_type][op_pos]["str"])
        return store[measure_type][op_pos]["res"]

    def dummy_measurement(self, measure_type, op_pos, clean=False, cnots=()):
        """Mark that a measurement is needed (placeholder entry)."""
        if cnots:
            if any(e for e in self.__c_measurements[measure_type]
                   if e["cnots"] == cnots and e["op_pos"] == op_pos):
                return
            self.__c_measurements[measure_type].append(
                {"cnots": cnots, "op_pos": op_pos, "data": 1,
                 "str": str(self.build_circuit(measure_type, op_pos, cnots))}
            )
        else:
            store = self.__clean_measurements if clean else self.__measurements
            if store[measure_type][op_pos] is None:
                store[measure_type][op_pos] = {
                    "res": 1,
                    "str": str(self.build_circuit(measure_type, op_pos, cnots))
                    if self.verbose else "",
                }

    # ---------- Helpers ----------
    def apply_full_hadamard(self):
        for a in range(self.n_qubits):
            self.m_state.h(a)

    def counts_to_prob(self, counts):
        res = zeros(1 << self.n_qubits)
        for bitstr, c in counts.items():
            res[int(bitstr, 2)] = c / self.n_shots
        return res

    def measure_state(self, circuit):
        """Run circuit and return probability distribution."""
        res = zeros(1 << self.m_state.num_qubits)
        if self.execution_type == qutils.execution_type.simulator:
            circuit.measure_all()
            counts = qutils.run_circuit(
                self.aer_sim, circuit, shots=self.n_shots)
            res = self.counts_to_prob(counts)
        elif self.execution_type == qutils.execution_type.statevector:
            sim = AerSimulator(method="statevector")
            circuit.save_statevector()
            statevector = asarray(
                sim.run(circuit).result().get_statevector(circuit))
            res = np.abs(statevector) ** 2
        return res

    def __len__(self):  # count stored measurements
        return (
            sum(1 for m in self.__measurements.values() for e in m if e)
            + sum(len(v) for v in self.__c_measurements.values())
            + sum(1 for m in self.__clean_measurements.values()
                  for e in m if e)
        )
