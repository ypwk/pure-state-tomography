import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator, StatevectorSimulator

import src.putils as putils
import src.qutils as qutils

MAX_CONC_JOB_COUNT = 3


class measurement_manager:
    def __init__(self, n_shots, execution_type, verbose: bool, batch_size: int = 1, partial_mixing: bool = False, noise_model=None, backend=None) -> None:
        self.n_shots = n_shots
        self.execution_type = execution_type
        self.verbose = verbose
        self.m_state = None
        self.clean_m_state = None
        self.partial_mixing = partial_mixing
        self.batch_size = batch_size

        self.num_measurements = 0

        # Sacred-compatible logging: just print if verbose=True
        self.verboseprint = print if verbose else (lambda *a, **k: None)

        self.__measurements = None
        self.__c_measurements = None
        self.__clean_measurements = None

        if self.execution_type == qutils.execution_type.simulator:
            if noise_model is not None and backend is not None:
                raise ValueError("Cannot set noise model and backend at the same time!!")
            if noise_model is not None:
                self.aer_sim = AerSimulator(
                    noise_model=noise_model, method="density_matrix", device="CPU"
                )
            if backend is not None:
                self.aer_sim = AerSimulator.from_backend(backend=backend)

        elif self.execution_type == qutils.execution_type.statevector:
            self.aer_sim = StatevectorSimulator(
                precision="double", device="CPU")

            self.verboseprint("Using statevector simulator.")
        else:
            self.verboseprint(
                f"Execution type {self.execution_type} is not supported.")
        
        # config = self.aer_sim.configuration()
        # print("\n=== Aer Backend Configuration ===")
        # for attr in dir(config):
        #     if not attr.startswith("_"):
        #         try:
        #             value = getattr(config, attr)
        #             print(f"{attr}: {value}")
        #         except Exception:
        #             pass

    # ---------- State setup ----------
    def set_state(self, tomography_type, state: np.ndarray | QuantumCircuit) -> None:
        """Sets the state for tomography (state or process)."""
        if isinstance(state, np.ndarray):
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
    def build_circuit(self, measure_type, op_pos: int, cnots=(), hadamards=(), clean=False):
        """Construct circuit for a given measurement type and operator position."""
        base = self.clean_m_state if clean else self.m_state
        qc = base.copy("execute")
        qc.barrier()

        # Insert CNOTs or Hadamards
        if self.partial_mixing:
            for loc in hadamards:
                qc.h(self.n_qubits - loc - 1)
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
            qc.unitary([[1/np.sqrt(2), 1j/np.sqrt(2)],
                        [1/np.sqrt(2), -1j/np.sqrt(2)]], q)
        elif measure_type == qutils.m_type.identity:
            qc.id(q)

        return qc

    # ---------- Measurements ----------
    def add_measurement(self, measure_type, op_pos=0, cnots=(), hadamards=(), clean=False):
        """Perform measurement and store results in appropriate registry."""
        qc = self.build_circuit(measure_type, op_pos, cnots, hadamards, clean)
        # print(f"qc: {qc}")
        res = self.measure_state(qc)
        self.num_measurements += 1
        entry = {"res": res, "str": str(qc) if self.verbose else "Not Verbose"}

        if cnots or hadamards:
            self.__c_measurements[measure_type].append(
                {"cnots": cnots, "hadamards": hadamards, "op_pos": op_pos,
                    "data": res, "str": entry["str"]}
            )
        elif clean:
            self.__clean_measurements[measure_type][op_pos] = entry
        else:
            self.__measurements[measure_type][op_pos] = entry
        return res

    def fetch(self, measure_type, op_pos=0, cnots=(), hadamards=(), clean=False):
        """Retrieve measurement, adding it if not already present."""
        if cnots:
            for e in self.__c_measurements[measure_type]:
                if e["cnots"] == cnots and e["op_pos"] == op_pos:
                    self.verboseprint(e["str"])
                    return e["data"]
            return self.add_measurement(measure_type, op_pos, cnots, hadamards)
        if hadamards:
            for e in self.__c_measurements[measure_type]:
                if e["hadamards"] == hadamards and e["op_pos"] == op_pos:
                    self.verboseprint(e["str"])
                    return e["data"]
            return self.add_measurement(measure_type, op_pos, cnots, hadamards)

        store = self.__clean_measurements if clean else self.__measurements
        if store[measure_type][op_pos] is None:
            return self.add_measurement(measure_type, op_pos, cnots=cnots, hadamards=hadamards, clean=clean)
        self.verboseprint(store[measure_type][op_pos]["str"])
        return store[measure_type][op_pos]["res"]

    # ---------- Helpers ----------
    def apply_full_hadamard(self):
        for a in range(self.n_qubits):
            self.m_state.h(a)

    def measure_state(self, circuit):
        """Run circuit batch_size times and return stacked probability distributions."""
        dim = 1 << self.m_state.num_qubits
        results = np.zeros((self.batch_size, dim))
        if self.execution_type == qutils.execution_type.simulator:
            circ = circuit.copy()
            circ.measure_all()
            results = qutils.run_circuit(
                self.aer_sim, circ, shots=self.n_shots, batch_size=self.batch_size)
        elif self.execution_type == qutils.execution_type.statevector:
            sim = AerSimulator(method="statevector")
            for i in range(self.batch_size):
                circ = circuit.copy()
                circ.save_statevector()
                statevector = np.asarray(
                    sim.run(circ).result().get_statevector(circ))
                results[i] = np.abs(statevector) ** 2
        # print(f"circuit: \n{circ}\n{circuit}")
        return results

    def __len__(self):  # count stored measurements
        return (
            sum(1 for m in self.__measurements.values() for e in m if e)
            + sum(len(v) for v in self.__c_measurements.values())
            + sum(1 for m in self.__clean_measurements.values()
                  for e in m if e)
        )
