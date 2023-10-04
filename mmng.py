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

from qiskit import transpile
from qiskit import Aer, execute
from qiskit_ibm_provider import IBMProvider

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
        self.__measurements[measure_type][op_pos] = res

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
        if measure_type == qutils.m_type.identity:
            return (
                self.add_cm(measure_type=measure_type, cnots=cnots, op_pos=op_pos)
                if self.__measurements[measure_type][0] is None
                else self.__measurements[measure_type][0]
            )
        return (
            self.add_cm(measure_type=measure_type, cnots=cnots, op_pos=op_pos)
            if self.__measurements[measure_type][op_pos] is None
            else self.__measurements[measure_type][op_pos]
        )

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

    def measure_state(self, circuit):
        """Measures a circuit using prior settings

        Args:
            circuit (qiskit.QuantumCircuit): The circuit to measure

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
                raw_result = qutils.run_circuit(circuit, self.simulator, shots=self.n_shots)
                for key in raw_result.keys():
                    res[int(key, 2)] = raw_result[key] / self.n_shots
        else:
            transpiled_circuit = transpile(circuit, self.device)
            job = execute(transpiled_circuit, backend=self.device, shots=self.n_shots)
            print(job.job_id())
            for key in job.result().get_counts():
                res[int(key, 2)] = raw_result[key] / self.n_shots

        return res


__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
