"""
This file contains utility code for managing quantum measurements and circuits.

Class:
- measurement_manager: Manages quantum measurements and circuits, allowing for
    measurement addition and fetching.

See each function's respective docstring for detailed usage and parameter information.
"""

from numpy import ndarray, sqrt, asarray, zeros

from datetime import datetime as dt

import os
import re

from qiskit import transpile, execute, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Session

import configparser
import putils
import qutils

MAX_CONC_JOB_COUNT = 3


class measurement_manager:
    # TODO: Implement main memory friendly version for applications at scale.

    def __init__(
        self, n_shots: int, execution_type: qutils.execution_type, verbose: bool
    ) -> None:
        """
        Initializes an instance of measurement manager, setting up the required
        parameters for quantum measurements and operations. It configures the
        quantum execution environment, verbosity settings, and manages internal
        state related to measurements.

        Args:
            n_shots (int): The number of measurement shots to be performed. This number
                        dictates how many times each quantum measurement is repeated.
            execution_type (qutils.execution_type): Specifies the type of execution
                        environment for the quantum operations (e.g., quantum processor,
                        simulator, statevector).
            verbose (bool): If set to True, enables detailed logging for the operations
                            performed by this instance. If False, logging is minimal.

        Attributes:
            n_shots (int): Stores the number of shots for quantum measurements.
            execution_type (qutils.execution_type): Holds the specified execution type.
            verbose (bool): Indicates whether verbose logging is enabled.
            m_state: Initially None, used to store the state of the quantum system.
            num_measurements (int): Counter for the number of measurements performed.
            my_job_file: Stores the job file path, if any.
            verbosefprint (function): A function that prints detailed information if
                                    verbose is True, else does nothing.
            verboseprint (function): Standard print function if verbose is True, else
                                     does nothing.
            __measurements: Private attribute to store measurement data.
            __c_measurements: Private attribute to store complex measurement data.
            device: Quantum device or backend retrieved based on the provided IBM token.
            session (Session): A session instance for interacting with the quantum
                backend.

        Raises:
            FileNotFoundError: If the 'config.ini' file is not found.
            KeyError: If the required IBM token is not found in the 'config.ini' file.

        Example usage:
            >>> quantum_instance = measurement_manager(n_shots=1000,
                                                execution_type=qutils.execution_type.ibm_qpu,
                                                verbose=True)

        Note:
            This constructor reads an API token from a 'config.ini' file to configure
            the IBM provider. Ensure that this file exists and is properly formatted
            with the necessary credentials.
        """
        self.n_shots = n_shots
        self.execution_type = execution_type
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
        self.device = provider.get_backend("ibm_kyoto")
        self.session = Session(backend=self.device)

    def set_state(
        self, tomography_type: qutils.tomography_type, state: ndarray | QuantumCircuit
    ) -> None:
        """Sets the state for this measurement manager."""
        if type(state) is ndarray:
            if tomography_type is qutils.tomography_type.state:
                putils.fprint("Input vector: {}".format(state), flush=True)
                self.n_qubits = putils.fast_log2(len(state))
                self.m_state = qutils.create_vector_circuit(state, self.n_qubits)
            elif tomography_type is qutils.tomography_type.process:
                if state.shape[0] != state.shape[1]:
                    raise Exception
                putils.fprint("Input matrix:\n{}".format(state), flush=True)
                self.n_qubits = putils.fast_log2(state.shape[0]) * 2
                self.m_state = qutils.create_matrix_circuit(state, self.n_qubits)
        elif type(state) is QuantumCircuit:
            if tomography_type is qutils.tomography_type.state:
                putils.fprint("Input circuit:\n{}".format(str(state)), flush=True)
                self.n_qubits = state.num_qubits
                self.m_state = state
                putils.fprint(
                    "Statevector:\n{}".format(
                        qutils.circuit_to_statevector(self.m_state)
                    )
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
                    if (
                        self.__measurements[t][op_pos] is not None
                        and type(self.__measurements[t][op_pos]["res"]) is int
                    ):
                        measurements += 1
                        try:
                            job_id = self.add_m(t, op_pos)
                        except Exception as error:
                            self.verbosefprint(error)
                            self.verbosefprint(
                                "Concurrent Job Limit Reached! Stopping..."
                            )
                            return
                        finally:
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
                        try:
                            job_id = self.add_cm(
                                t,
                                self.__c_measurements[t][cm]["cnots"],
                                self.__c_measurements[t][cm]["op_pos"],
                            )
                        except Exception as error:
                            self.verbosefprint(error)
                            self.verbosefprint(
                                "Concurrent Job Limit reached! Stopping..."
                            )
                        finally:
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
        """Constructs a quantum circuit based on the specified measurements and
        operations.

        Args:
            measure_type (qutils.m_type): Type of measurement to be performed.
            op_pos (int): Position of the operation in the circuit.
            cnots (list, optional): List of positions where CNOT gates should be
                                    inserted. Defaults to [None], implying no CNOT
                                    gates are to be inserted.

        Returns:
            state_circuit (qiskit.circuit.QuantumCircuit): The constructed quantum
            circuit.
        """
        state_circuit = self.m_state.copy("execute")
        state_circuit.barrier()

        if len(cnots) > 0:
            for a in cnots:
                if a is not None:
                    state_circuit.cx(
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
                cnots = [item for item in data[3].split(",") if item != ""]
                if len(cnots) == 0:
                    cnots = [None]
                else:
                    cnots = [
                        [
                            int(re.sub("[^0-9]", "", cnots[i])),
                            int(re.sub("[^0-9]", "", cnots[i + 1])),
                        ]
                        for i in range(0, len(cnots), 2)
                    ]

                if self.verbose:
                    state_circuit = self.construct_circuit(
                        measure_type=m, op_pos=op_pos, cnots=cnots
                    )

                if cnots[0] is None:
                    res = zeros(putils.fast_pow(2, self.m_state.num_qubits))
                    for key in run_data.keys():
                        res[int(key, 16)] = run_data[key] / self.n_shots
                    self.__measurements[m][op_pos] = {
                        "res": res,
                        "str": None if self.verbose is False else str(state_circuit),
                    }
                else:
                    res = zeros(putils.fast_pow(2, self.m_state.num_qubits))
                    for key in run_data.keys():
                        res[int(key, 16)] = run_data[key] / self.n_shots
                    self.__c_measurements[m].append(
                        {
                            "cnots": cnots,
                            "op_pos": op_pos,
                            "data": res,
                            "str": None
                            if self.verbose is False
                            else str(state_circuit),
                        }
                    )

        return len(lines)

    def add_cm(self, measure_type, cnots, op_pos=0) -> ndarray:
        """Carries out a measurement on the pure state accounting for CNOT operations,
        returning the correct value.

        Args:
            measure_type (qutils.m_type): The type of operator used in the tensor
                structure.
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
                "str": "Not Verbose" if self.verbose is False else str(state_circuit),
            }
        )

        return res

    def add_m(self, measure_type, op_pos=0) -> ndarray:
        """Carries out a measurement on the pure state, returning the correct value.

        Args:
            measure_type (qutils.m_type): The type of operator used in the tensor
                structure
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
        """Fetches a measurement result, denoted by location of and whether the
        Hadamard/alternate operator has been inverted in the tensor structure.
        Also considers CNOT placement.

        Args:
            measure_type (qutils.m_type): The type of operator used in the tensor
                structure
            cnots (list): The position of CNOTs in the tensor structure
            op_pos (int): The position of the Hadamard in the tensor product, zero
                indexed from the left.

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
        """Fetches a measurement result, denoted by location of and whether the
        Hadamard/alternate operator has been inverted in the tensor structure.

        Args:
            measure_type (qutils.m_type): The type of operator used in the tensor
                structure
            op_pos (int): The position of the Hadamard in the tensor product, zero
                indexed from the left.

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
            measure_type (qutils.m_type): The type of operator used in the tensor
                structure
            cnots (list): The position of CNOTs in the tensor structure
            op_pos (int): The position of the Hadamard in the tensor product, zero
                indexed from the left.
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
                if entry["cnots"] == cnots and entry["op_pos"] == op_pos:
                    self.verbosefprint(entry["str"])
                    return

            if self.verbose:
                state_circuit = self.construct_circuit(
                    measure_type=measure_type, op_pos=op_pos, cnots=cnots
                )

            self.__c_measurements[measure_type].append(
                {
                    "cnots": cnots,
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

        if self.execution_type == qutils.execution_type.simulator:
            # Shot-based simulation using AerSimulator
            circuit.measure_all()
            raw_result = qutils.run_circuit(
                circuit, shots=self.n_shots, backend=self.device
            )
            for key in raw_result.keys():
                res[int(key, 2)] = raw_result[key] / self.n_shots
        elif self.execution_type == qutils.execution_type.statevector:
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
        elif self.execution_type == qutils.execution_type.ibm_qpu:
            circuit.measure_all()
            transpiled_circuit = transpile(circuit, self.device)
            res = execute(
                transpiled_circuit, backend=self.device, shots=self.n_shots
            ).job_id()
        return res


__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
