"""
This file contains utility code for managing quantum measurements and circuits.

Classes and Functions:
- meas_manager: Manages quantum measurements and circuits, allowing for measurement addition and fetching.
- add_cm: Carries out a measurement on the pure state accounting for CNOT operations, returning the correct value.
- add_m: Carries out a measurement on the pure state, returning the correct value.
- fetch_cm: Fetches a measurement result considering CNOT placement and whether the Hadamard/alternate operator is
            inverted.
- fetch_m: Fetches a measurement result considering whether the Hadamard/alternate operator is inverted.
- measure_state: Measures a circuit using prior settings.

See each function's respective docstring for detailed usage and parameter information.
"""

from numpy import ndarray, sqrt, asarray, zeros

from datetime import datetime as dt

import os

from qiskit import transpile
from qiskit import execute
from qiskit_aer import AerSimulator
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Session

import configparser
import putils
import qutils

MAX_CONC_JOB_COUNT = 3


class meas_manager:
    # TODO: Implement main memory friendly version for applications at exponential scale.

    def __init__(
        self, n_shots, simulator, use_statevector, verbose
    ) -> None:
        self.n_shots = n_shots
        self.simulator = simulator
        self.use_statevector = use_statevector
        self.verbose = verbose
        self.m_state = None

        self.num_measurements = 0
        self.my_job_file = None

        self.verbosefprint = putils.fprint if verbose else lambda *a, **k: None
        self.verboseprint = print if verbose else lambda *a, **k: None

        self.__measurements = None
        self.__c_measurements = None

        api_token = ""
        with open("config.ini", "r") as cf:
            cp = configparser.ConfigParser()
            cp.read_file(cf)
            api_token = cp.get("IBM", "token")

        provider = IBMProvider(token=api_token)
        self.device = provider.get_backend("ibm_lagos")
        self.session = Session(backend=self.device)

    def set_state(self, state) -> None:
        """Sets the state for this measurement manager."""

        if state.ndim == 1:
            self.verbosefprint("Input vector: {}".format(state), flush=True)
            self.n_qubits = putils.fast_log2(len(state))
            self.m_state = qutils.create_vector_circuit(state, self.n_qubits)
        else:
            if state.shape[0] != state.shape[1]:
                raise Exception
            self.verbosefprint("Input matrix:\n{}".format(state), flush=True)
            self.n_qubits = putils.fast_log2(state.shape[0]) * 2
            self.m_state = qutils.create_matrix_circuit(state, self.n_qubits)

        self.__measurements = {
            qutils.m_type.identity: [None for _ in range(self.n_qubits)],
            qutils.m_type.cmplx_hadamard: [None for _ in range(self.n_qubits)],
            qutils.m_type.real_hadamard: [None for _ in range(self.n_qubits)],
        }
        self.__c_measurements = {
            qutils.m_type.identity: [],
            qutils.m_type.cmplx_hadamard: [],
            qutils.m_type.real_hadamard: [],
        }
        self.num_measurements = 0

    def to_job_file(self) -> None:
        """Requests for measurements and records job IDs in a job file."""
        if self.my_job_file is None:
            self.my_job_file = "job_{}.txt".format(
                dt.now().strftime("%Y_%m_%dT_%H_%M_%S")
            )
            self.verboseprint(
                "Warning: this program has not read from a job file yet, printing to",
                self.my_job_file,
            )
            self.verbosefprint("Corresponding job file:", self.my_job_file, flush=True)

        job_path = os.path.join("jobs", self.my_job_file)
        measurements = 0

        if not os.path.isdir("jobs"):
            os.mkdir("jobs")

        with open(job_path, mode="a") as f:
            for t in qutils.m_type:
                for op_pos in range(len(self.__measurements[t])):
                    if type(self.__measurements[t][op_pos]) is int:
                        measurements += 1
                        job_id = self.add_m(t, op_pos)
                        f.write(
                            "{}:{}:{}:{}\n".format(
                                job_id,
                                t,
                                op_pos,
                                ",".join(str(_) for _ in []),
                            )
                        )
                        if measurements == MAX_CONC_JOB_COUNT:
                            return
                for cm in range(len(self.__c_measurements[t])):
                    if type(self.__c_measurements[t][cm]["data"]) is int:
                        measurements += 1
                        job_id = self.add_cm(
                            t,
                            self.__c_measurements[t][cm]["cnots"],
                            self.__c_measurements[t][cm]["op_pos"],
                        )
                        f.write(
                            "{}:{}:{}:{}\n".format(
                                job_id,
                                t,
                                self.__c_measurements[t][cm]["op_pos"],
                                ",".join(
                                    str(_)
                                    for _ in self.__c_measurements[t][cm]["cnots"]
                                ),
                            )
                        )
                        if measurements == MAX_CONC_JOB_COUNT:
                            return

    def construct_circuit(self, measure_type: qutils.m_type, op_pos: int, cnots=()):
        """Constructs a quantum circuit based on the specified measurements and operations.

        Args:
            measure_type (qutils.m_type): Type of measurement to be performed.
            op_pos (int): Position of the operation in the circuit.
            cnots (list, optional): List of positions where CNOT gates should be inserted.
                                    Defaults to [None], implying no CNOT gates are to be inserted.

        Returns:
            state_circuit (qiskit.circuit.QuantumCircuit): The constructed quantum circuit.
        """
        state_circuit = self.m_state.copy("execute")
        state_circuit.barrier()

        if len(cnots) > 0:
            for a in cnots:
                state_circuit.cnot(
                    self.m_state.num_qubits - a[0] - 1,
                    self.m_state.num_qubits - a[1] - 1,
                )
            state_circuit.barrier()

        for a in range(self.m_state.num_qubits):
            if a == self.m_state.num_qubits - op_pos - 1:
                if measure_type == qutils.m_type.real_hadamard:
                    state_circuit.h(a)
                elif measure_type == qutils.m_type.cmplx_hadamard:
                    state_circuit.unitary(
                        [
                            [1 / sqrt(2), 1j / sqrt(2)],
                            [1 / sqrt(2), -1j / sqrt(2)],
                        ],
                        a,
                    )
                elif measure_type == qutils.m_type.identity:
                    state_circuit.i(a)
            else:
                state_circuit.i(a)

        return state_circuit

    def consume_job_file(self, fname) -> bool:
        """Consumes job file, fetching and refactoring measurement results.

        Args:
            fname (str): Name of job file.
        Returns:
            res (bool): Number of measurements in job file.
        """
        self.my_job_file = fname

        job_path = os.path.join("jobs", self.my_job_file)

        service = QiskitRuntimeService()
        with open(job_path) as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(":")
                job = service.job(job_id=data[0])
                run_data = job.result().data()["counts"]

                m = qutils.m_type.identity
                if "real" in data[1]:
                    m = qutils.m_type.real_hadamard
                elif "cmplx" in data[1]:
                    m = qutils.m_type.cmplx_hadamard

                op_pos = int(data[2])

                cnots = [int(i) if i != "" else None for i in list(data[3].split(","))]

                if self.verbose:
                    state_circuit = self.construct_circuit(
                        measure_type=m, op_pos=op_pos, cnots=cnots
                    )

                if cnots[0] is None:
                    res = zeros(putils.fast_pow(2, self.m_state.num_qubits))
                    for key in run_data.keys():
                        res[int(key[2:])] = run_data[key] / self.n_shots
                    self.__measurements[m][op_pos] = {
                        "res": res,
                        "str": None if self.verbose is False else str(state_circuit),
                    }
                else:
                    res = zeros(putils.fast_pow(2, self.m_state.num_qubits))
                    for key in run_data.keys():
                        res[int(key[2:])] = run_data[key] / self.n_shots
                    self.__c_measurements[m].append(
                        {
                            "cnots": set(cnots),
                            "op_pos": op_pos,
                            "data": res,
                            "str": None
                            if self.verbose is False
                            else str(state_circuit),
                        }
                    )

        return len(lines)

    def add_cm(self, measure_type, cnots, op_pos=0) -> ndarray:
        """Carries out a measurement on the pure state accounting for CNOT operations, returning the correct value.

        Args:
            measure_type (qutils.m_type): The type of operator used in the tensor structure
            cnots (list): The position of CNOT measurements in circuit
            op_pos (int, optional): The operation of the position of the operator
                    Defaults to 0, since it is unused when the operator is the identity
        """

        if measure_type not in [
            qutils.m_type.cmplx_hadamard,
            qutils.m_type.real_hadamard,
            qutils.m_type.identity,
        ]:
            raise ValueError("Measurement type {} not valid.", measure_type)

        if len(cnots) == 0:
            return self.add_m(measure_type, op_pos)

        state_circuit = self.construct_circuit(
            measure_type=measure_type, op_pos=op_pos, cnots=cnots
        )

        res = self.measure_state(state_circuit)

        self.num_measurements += 1
        self.__c_measurements[measure_type].append(
            {
                "cnots": cnots,
                "op_pos": op_pos,
                "data": res,
                "str": None if self.verbose is False else str(state_circuit),
            }
        )

        return res

    def add_m(self, measure_type, op_pos=0) -> ndarray:
        """Carries out a measurement on the pure state, returning the correct value.

        Args:
            measure_type (qutils.m_type): The type of operator used in the tensor structure
            op_pos (int, optional): The operation of the position of the operator
                    Defaults to 0, since it is unused when the operator is the identity
        """

        if measure_type not in [
            qutils.m_type.cmplx_hadamard,
            qutils.m_type.real_hadamard,
            qutils.m_type.identity,
        ]:
            raise ValueError("Measurement type {} not valid.", measure_type)

        state_circuit = self.construct_circuit(measure_type=measure_type, op_pos=op_pos)

        res = self.measure_state(state_circuit)

        self.num_measurements += 1
        self.__measurements[measure_type][op_pos] = {
            "res": res,
            "str": None if self.verbose is False else str(state_circuit),
        }

        return res

    def fetch_cm(self, measure_type, cnots, op_pos) -> ndarray:
        """Fetches a measurement result, denoted by location of and whether the Hadamard/alternate operator has been
        inverted in the tensor structure.
        Also considers CNOT placement/

        Args:
            measure_type (qutils.m_type): The type of operator used in the tensor structure
            cnots (list): The position of CNOTs in the tensor structure
            op_pos (int): The position of the Hadamard in the tensor product, zero indexed from the left.

        Returns:
            measurement_result (numpy.ndarray):
        """
        for entry in self.__c_measurements[measure_type]:
            if entry["cnots"] == cnots and entry["op_pos"] == op_pos:
                self.verbosefprint(entry["str"])
                return entry["data"]
        measurement_result = self.add_cm(
            measure_type=measure_type, cnots=cnots, op_pos=op_pos
        )
        self.verbosefprint(self.__c_measurements[measure_type][-1]["str"])

        return measurement_result

    def fetch_m(self, measure_type, op_pos) -> ndarray:
        """Fetches a measurement result, denoted by location of and whether the Hadamard/alternate operator has been
        inverted in the tensor structure.

        Args:
            measure_type (qutils.m_type): The type of operator used in the tensor structure
            op_pos (int): The position of the Hadamard in the tensor product, zero indexed from the left.

        Returns:
            measurement_result (numpy.ndarray):
        """
        if measure_type == qutils.m_type.identity:
            if self.__measurements[measure_type][0] is None:
                measurement_result = self.add_m(
                    measure_type=measure_type, op_pos=op_pos
                )
            else:
                measurement_result = self.__measurements[measure_type][0]["res"]
            self.verbosefprint(self.__measurements[measure_type][0]["str"])
            return measurement_result

        if self.__measurements[measure_type][op_pos] is None:
            measurement_result = self.add_m(measure_type=measure_type, op_pos=op_pos)
        else:
            measurement_result = self.__measurements[measure_type][op_pos]["res"]
        self.verbosefprint(self.__measurements[measure_type][op_pos]["str"])
        return measurement_result

    def dummy_measurement(self, measure_type, op_pos, cnots=[]) -> None:
        """Stores that a measurement is necessary for this specific configuration.

        Args:
            measure_type (qutils.m_type): The type of operator used in the tensor structure
            cnots (list): The position of CNOTs in the tensor structure
            op_pos (int): The position of the Hadamard in the tensor product, zero indexed from the left.
        """
        if len(cnots) == 0:
            if self.__measurements[measure_type][op_pos] is not None:
                self.verbosefprint(self.__measurements[measure_type][op_pos]["str"])
                return
            else:
                if self.verbose:
                    state_circuit = self.construct_circuit(
                        measure_type=measure_type, op_pos=op_pos, cnots=cnots
                    )

                self.__measurements[measure_type][op_pos] = {
                    "res": 1,
                    "str": str(state_circuit),
                }
                self.verbosefprint(self.__measurements[measure_type][op_pos]["str"])
        else:
            for entry in self.__c_measurements[measure_type]:
                if entry["cnots"] == set(cnots) and entry["op_pos"] == op_pos:
                    self.verbosefprint(entry["str"])
                    return

            if self.verbose:
                state_circuit = self.construct_circuit(
                    measure_type=measure_type, op_pos=op_pos, cnots=cnots
                )

            self.__c_measurements[measure_type].append(
                {
                    "cnots": set(cnots),
                    "op_pos": op_pos,
                    "data": 1,
                    "str": str(state_circuit),
                }
            )
            self.verbosefprint(self.__c_measurements[measure_type][-1]["str"])

    def __len__(self) -> int:
        """Counts the number of measurements stored in the measurement manager.

        Returns:
            int: number of stored measurements
        """
        ret = 0
        for m in [
            qutils.m_type.identity,
            qutils.m_type.cmplx_hadamard,
            qutils.m_type.real_hadamard,
        ]:
            for a in range(len(self.__measurements[m])):
                if self.__measurements[m][a] is not None:
                    ret += 1
            ret += len(self.__c_measurements[m])
        return ret

    def measure_state(self, circuit):
        """Measures a circuit using prior settings

        Args:
            circuit (qiskit.QuantumCircuit): The circuit to measure.

        Returns:
            measurement_result (numpy.ndarray):
        """
        res = zeros(putils.fast_pow(2, self.m_state.num_qubits))

        if self.simulator:
            if self.use_statevector:
                # using statevector for precise execution
                simulator = AerSimulator(method="statevector")
                circuit.save_statevector()
                raw_result = simulator.run(circuit).result()
                statevector = asarray(raw_result.get_statevector(circuit))
                for idx in range(len(statevector)):
                    res[idx] = abs(
                        statevector[idx].real * statevector[idx].real
                        + statevector[idx].imag * statevector[idx].imag
                    )
            else:
                # Shot-based simulation using AerSimulator
                circuit.measure_all()
                raw_result = qutils.run_circuit(
                    circuit, self.simulator, shots=self.n_shots, backend=self.device
                )
                for key in raw_result.keys():
                    res[int(key, 2)] = raw_result[key] / self.n_shots
        else:
            circuit.measure_all()
            transpiled_circuit = transpile(circuit, self.device)
            res = execute(
                transpiled_circuit, backend=self.device, shots=self.n_shots
            ).job_id()
        return res


__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
