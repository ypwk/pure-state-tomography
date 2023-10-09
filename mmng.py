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

from qiskit import transpile
from qiskit import Aer, execute
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService

import configparser
import putils
import qutils


class meas_manager:
    # TODO: Implement main memory friendly version for applications at exponential scale.

    def __init__(
        self, m_state, n_qubits, n_shots, simulator, use_statevector, verbose
    ) -> None:
        self.n_qubits = n_qubits
        self.m_state = m_state
        self.num_measurements = 0
        self.n_shots = n_shots
        self.simulator = simulator
        self.use_statevector = use_statevector
        self.__measurements = {
            qutils.m_type.identity: [None for _ in range(n_qubits)],
            qutils.m_type.cmplx_hadamard: [None for _ in range(n_qubits)],
            qutils.m_type.real_hadamard: [None for _ in range(n_qubits)],
        }
        self.__c_measurements = {
            qutils.m_type.identity: [],
            qutils.m_type.cmplx_hadamard: [],
            qutils.m_type.real_hadamard: [],
        }
        self.verbose = verbose

        if not self.simulator:
            api_token = ""
            with open("config.ini", "r") as cf:
                cp = configparser.ConfigParser()
                cp.read_file(cf)
                api_token = cp.get("IBM", "token")
            n_qubits = 2

            provider = IBMProvider(token=api_token)
            self.device = provider.get_backend("ibm_lagos")

    def to_job_file(self) -> None:
        """Requests for measurements and records job IDs in a job file."""
        fn = "job_{}.txt".format(dt.now().strftime("%Y_%m_%dT_%H_%M_%S"))
        putils.fprint("Corresponding job file:", fn, flush=True)
        with open(
            fn, mode="w"
        ) as f:
            for type in qutils.m_type:
                for op_pos in range(len(self.__measurements[type])):
                    if self.__measurements[type][op_pos] is not None:
                        for cnots in self.__measurements[type][op_pos]:
                            job_id = self.add_cm(type, list(cnots), op_pos)
                            f.write(
                                "{}:{}:{}:{}\n".format(
                                    job_id,
                                    type,
                                    op_pos,
                                    ",".join(str(_) for _ in cnots),
                                )
                            )

    def consume_job_file(self, fname) -> bool:
        """Consumes job file, fetching and refactoring measurement results.

        Args:
            fname (str): Name of job file.
        Returns:
            res (bool): Whether or not to do a dry run or an actual run for inference.
        """
          
        service = QiskitRuntimeService()
        with open(fname) as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(":")
                job = service.job(job_id=data[0])
   
                print(job.result().data())

                m = qutils.m_type.identity
                if "real" in data[1]:
                    m = qutils.m_type.real_hadamard
                elif "cmplx" in data[1]:
                    m = qutils.m_type.cmplx_hadamard

                op_pos = int(data[2])

                cnots = [int(i) if i != "" else None for i in list(data[3].split(","))]
                if cnots[0] is None:
                    res = zeros(putils.fast_pow(2, self.m_state.num_qubits))
                    for key in job.result().keys():
                        res[int(key, 2)] = job.result()[key] / self.n_shots
                    self.__measurements[m][op_pos] = res
                else:
                    res = zeros(putils.fast_pow(2, self.m_state.num_qubits))
                    for key in job.result().keys():
                        res[int(key, 2)] = job.result()[key] / self.n_shots
                    self.__c_measurements[m].append({"cnots": set(cnots), "data": res})

        return m == qutils.m_type.identity and len(lines) == 1

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

        state_circuit = self.m_state.copy("execute")
        for a in range(self.m_state.num_qubits):
            if self.m_state.num_qubits - a - 1 in cnots:
                state_circuit.cnot(
                    self.m_state.num_qubits - a - 1, self.m_state.num_qubits - a - 2
                )
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

        res = self.measure_state(state_circuit)

        self.num_measurements += 1
        self.__c_measurements[measure_type].append({"cnots": set(cnots), "data": res})

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

        state_circuit = self.m_state.copy("execute")
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

        res = self.measure_state(state_circuit)

        self.num_measurements += 1
        self.__measurements[measure_type][op_pos] = res

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
            if entry["cnots"] == set(cnots):
                return entry["data"]
        return self.add_cm(measure_type=measure_type, cnots=cnots, op_pos=op_pos)

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
            return (
                self.add_m(measure_type=measure_type, op_pos=op_pos)
                if self.__measurements[measure_type][0] is None
                else self.__measurements[measure_type][0]
            )
        return (
            self.add_m(measure_type=measure_type, op_pos=op_pos)
            if self.__measurements[measure_type][op_pos] is None
            else self.__measurements[measure_type][op_pos]
        )

    def dummy_measurement(self, measure_type, op_pos, cnots=[]) -> None:
        """Stores that a measurement is necessary for this specific configuration.

        Args:
            measure_type (qutils.m_type): The type of operator used in the tensor structure
            cnots (list): The position of CNOTs in the tensor structure
            op_pos (int): The position of the Hadamard in the tensor product, zero indexed from the left.
        """
        cnots = set(cnots)
        if self.__measurements[measure_type][op_pos] is not None:
            for m in self.__measurements[measure_type][op_pos]:
                if m == cnots:
                    return
        else:
            self.__measurements[measure_type][op_pos] = []
        self.__measurements[measure_type][op_pos].append(cnots)

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
                simulator = Aer.get_backend("statevector_simulator")
                raw_result = execute(circuit, simulator).result()
                statevector = asarray(raw_result.get_statevector(circuit))
                for idx in range(len(statevector)):
                    res[idx] = abs(
                        statevector[idx].real * statevector[idx].real
                        - statevector[idx].imag * statevector[idx].imag
                    )
            else:
                circuit.measure_all()
                raw_result = qutils.run_circuit(
                    circuit, self.simulator, shots=self.n_shots
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
