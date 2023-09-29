#!/usr/bin/env python

print("Initializing . . .", end="", flush=True)

from numpy import ndarray, array, sqrt, asarray, zeros, linalg
import numpy.random as nprandom

from qiskit import QuantumCircuit, transpile, result, assemble
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import Aer, execute

import networkx as nx

from enum import Enum


class m_type(Enum):
    real_hadamard = 1
    cmplx_hadamard = 2
    identity = 3


def fast_pow(base, exp) -> int:
    """Fast exponent using squaring

    Args:
        base (int): base of exponent
        exp (int): power of exponent

    Returns:
        result: result of exponent
    """
    res = 1
    while True:
        if exp & 1:
            res *= base
        exp = exp >> 1
        if not exp:
            break
        base *= base
    return res


def fast_log2(num: int) -> int:
    """Fast log2 using bit length

    Args:
        num (int): the number to log

    Returns:
        int: the result
    """
    return num.bit_length() - 1


def hamming(i1: int, i2: int) -> int:
    """Computes the Hamming distance between two integers

    Args:
        i1 (int): integer 1
        i2 (int): integer 2

    Returns:
        int: The Hamming distance between the two integers
    """
    x = i1 ^ i2
    ones = 0

    while x > 0:
        ones += x & 1
        x >>= 1

    return ones


def infer_target(target_idx, source_idx, source_val, h_measure, v_measure) -> ndarray:
    """Calculates and returns the value of an entry using previously inferred values in the measurement results.

    Args:
        target_idx (int): The index of the target value to infer
        source_idx (int): The index of the value to use to infer the target
        source_val (numpy.ndarray): The source value
        h_measure (numpy.ndarray): The array of measurements with the Hadamard gate
        v_measure (numpy.ndarray): The array of measurements with the alternate gate

    Returns: numpy.ndarray
    """

    res = array([0.0, 0.0])
    if target_idx < source_idx:  # backwards
        res[0] = (
            source_val[1] * (v_measure[source_idx] - v_measure[target_idx])
            + source_val[0] * (h_measure[target_idx] - h_measure[source_idx])
        ) / (2 * (source_val[0] * source_val[0] + source_val[1] * source_val[1]))

        res[1] = (
            source_val[0] * (v_measure[source_idx] - v_measure[target_idx])
            - source_val[1] * (h_measure[target_idx] - h_measure[source_idx])
        ) / (2 * (source_val[0] * source_val[0] + source_val[1] * source_val[1]))

    else:  # forwards
        res[0] = (
            source_val[1] * (v_measure[target_idx] - v_measure[source_idx])
            + source_val[0] * (h_measure[source_idx] - h_measure[target_idx])
        ) / (2 * (source_val[0] * source_val[0] - source_val[1] * source_val[1]))

        res[1] = (
            source_val[1] * (h_measure[source_idx] - h_measure[target_idx])
            + source_val[0] * (v_measure[target_idx] - v_measure[source_idx])
        ) / (2 * (source_val[0] * source_val[0] + source_val[1] * source_val[1]))

    return res


def create_circuit(state, n_qubits) -> QuantumCircuit:
    """Initializes a state as a qiskit QuantumCircuit

    Args:
        state (numpy.ndarray): The state to initialize
        n_qubits (int): The number of qubits used to represent the staet

    Returns:
        qiskit.QuantumCircuit:
    """
    qc = QuantumCircuit(n_qubits)
    qc.initialize(state, [_ for _ in range(n_qubits)])
    return qc


def run_circuit(qc, simulator, shots=1024) -> result.counts.Counts:
    """Runs the circuit on the simulator

    Args:
        qc (qiskit.QuantumCircuit): Quantum circuit to run
        simulator (bool): Whether or not to use a simulator
        shots (int): Number of shots to take

    Returns:
        numpy.ndarray: An array of result counts
    """
    if simulator:
        aer_sim = Aer.get_backend("aer_simulator")
        t_qc = transpile(qc, aer_sim)
        result = aer_sim.run(t_qc, shots=shots).result()
        counts = result.get_counts(qc)
    else:
        counts = zeros([0, 0, 0, 0])
    return counts


def find_nonzero_positions(counts, epsilon=1e-5) -> list:
    """Finds positions with nonzero counts in the counts array

    Args:
        counts (numpy.ndarray): array filled with counts

    Returns:
        List: a list of counts
    """
    positions = []
    for c in range(len(counts)):
        if counts[c] > epsilon:
            positions.append(c)
    return positions


class meas_manager:
    # TODO: Implement main memory friendly version for applications at exponential scale.
    
    def __init__(self, m_state, n_qubits, n_shots, simulator, use_statevector, verbose) -> None:
        self.n_qubits = n_qubits
        self.m_state = m_state
        self.num_measurements = 0
        self.n_shots = n_shots
        self.simulator = simulator
        self.use_statevector = use_statevector
        self.__measurements = {
            m_type.identity: [None for _ in range(n_qubits)],
            m_type.cmplx_hadamard: [None for _ in range(n_qubits)],
            m_type.real_hadamard: [None for _ in range(n_qubits)],
        }
        self.verbose = verbose
        
        if not self.simulator:
            self.service = QiskitRuntimeService()

    def add_cm(self, measure_type, cnots, op_pos=0) -> ndarray:
        """Carries out a measurement on the pure state accounting for CNOT operations, returning the correct value.

        Args:
            measure_type (m_type): The type of operator used in the tensor structure
            cnots (list): The position of CNOT measurements in circuit
            op_pos (int, optional): The operation of the position of the operator
                    Defaults to 0, since it is unused when the operator is the identity
        """

        if measure_type not in [
            m_type.cmplx_hadamard,
            m_type.real_hadamard,
            m_type.identity,
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
                if measure_type == m_type.real_hadamard:
                    state_circuit.h(a)
                elif measure_type == m_type.cmplx_hadamard:
                    state_circuit.unitary(
                        [
                            [1 / sqrt(2), 1j / sqrt(2)],
                            [1 / sqrt(2), -1j / sqrt(2)],
                        ],
                        a,
                    )
                elif measure_type == m_type.identity:
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
            measure_type (m_type): The type of operator used in the tensor structure
            op_pos (int, optional): The operation of the position of the operator
                    Defaults to 0, since it is unused when the operator is the identity
        """

        if measure_type not in [
            m_type.cmplx_hadamard,
            m_type.real_hadamard,
            m_type.identity,
        ]:
            raise ValueError("Measurement type {} not valid.", measure_type)

        state_circuit = self.m_state.copy("execute")
        for a in range(self.m_state.num_qubits):
            if a == self.m_state.num_qubits - op_pos - 1:
                if measure_type == m_type.real_hadamard:
                    state_circuit.h(a)
                elif measure_type == m_type.cmplx_hadamard:
                    state_circuit.unitary(
                        [
                            [1 / sqrt(2), 1j / sqrt(2)],
                            [1 / sqrt(2), -1j / sqrt(2)],
                        ],
                        a,
                    )
                elif measure_type == m_type.identity:
                    state_circuit.i(a)
            else:
                state_circuit.i(a)

        res = self.measure_state(state_circuit)

        self.num_measurements += 1
        self.__measurements[measure_type][op_pos] = res

        return res

    def fetch_cm(self, measure_type, cnots, op_pos) -> ndarray:
        """Fetches a measurement result, denoted by location of and whether the Hadamard/alternate operator has been inverted in the tensor structure.
        Also considers CNOT placement/

        Args:
            measure_type (m_type): The type of operator used in the tensor structure
            cnots (list): The position of CNOTs in the tensor structure
            op_pos (int): The position of the Hadamard in the tensor product, zero indexed from the left.

        Returns:
            measurement_result (numpy.ndarray):
        """
        if measure_type == m_type.identity:
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
        """Fetches a measurement result, denoted by location of and whether the Hadamard/alternate operator has been inverted in the tensor structure.

        Args:
            measure_type (m_type): The type of operator used in the tensor structure
            op_pos (int): The position of the Hadamard in the tensor product, zero indexed from the left.

        Returns:
            measurement_result (numpy.ndarray):
        """
        if measure_type == m_type.identity:
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
        res = zeros(fast_pow(2, self.m_state.num_qubits))
        
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
                raw_result = run_circuit(circuit, self.simulator, shots=self.n_shots)
                for key in raw_result.keys():
                    res[int(key, 2)] = raw_result[key] / self.n_shots
        else: 
            service.backends(simulator=False, operational=True, min_num_qubits=5) #TODO set up simulator
                
        return res


def pure_state_tomography(input_state=None, n_qubits=2, precise=False, simulator=True, n_shots=8192, verbose=False):
    """Uses the pure state tomography algorithm to infer the state of a hidden input_vector

    Args:
        input_state (numpy.ndarray, optional): The hidden input_vector to infer.
        n_qubits (int, optional): The number of qubits needed to represent the input_vector. Defaults to 2.
        precise (bool): Whether or not to use precise measurements on Qiskit
        simulator (bool): Whether or not to use Qiskit simulator or IBM quantum machine
        n_shots (int): The number of shots to use for imprecise measurements on Qiskit. Only comes into effect when precise is False. 
        verbose (bool): Whether to print detailed information
    """

    DIM = fast_pow(2, n_qubits)
    if input_state is None:  # generate random input vector if it is None
        input_state = nprandom.rand(DIM) + 1j * nprandom.rand(DIM)
        input_state = input_state / linalg.norm(input_state)
        
    print("Input vector: {}".format(input_state), flush=True)

    mystery_state = create_circuit(input_state, n_qubits)

    # do initial identity measurement
    mm = meas_manager(
        m_state=mystery_state, n_qubits=n_qubits, n_shots=n_shots, simulator=simulator, use_statevector=precise, verbose=verbose
    )

    res = zeros((DIM, 2))
    __iter_inf_helper(res, mm)

    print("Number of unitary operators applied: {}".format(mm.num_measurements))

    return [res[a][0] + 1j * res[a][1] for a in range(DIM)]


def __iter_inf_helper(
    target_arr: ndarray,
    mm: meas_manager,
) -> None:
    """An iterative implementation of the inference helper

    Args:
        target_arr (numpy.ndarray): The array with incomplete measurements
        mm (meas_manager): Measure manager keeping track of measurement values.
    """

    # do identity measurement to seed
    id_m = mm.add_m(m_type.identity, 0)
    counts = find_nonzero_positions(id_m)

    target_arr[counts[0]][0] = sqrt(id_m[0])
    t_list = set(counts)
    t_list.remove(counts[0])

    counts = set(counts)

    for op in range(mm.n_qubits):
        to_rem = set()
        for target in t_list:
            if target & (1 << (mm.n_qubits - op - 1)):
                if (
                    target - fast_pow(2, mm.n_qubits - op - 1) in counts
                    and target - fast_pow(2, mm.n_qubits - op - 1) not in t_list
                ):
                    cmplx_m = mm.fetch_m(measure_type=m_type.cmplx_hadamard, op_pos=op)
                    real_m = mm.fetch_m(measure_type=m_type.real_hadamard, op_pos=op)

                    target_arr[target] = infer_target(
                        target_idx=target,
                        source_idx=target - fast_pow(2, mm.n_qubits - op - 1),
                        source_val=target_arr[
                            target - fast_pow(2, mm.n_qubits - op - 1)
                        ],
                        h_measure=real_m,
                        v_measure=cmplx_m,
                    )
                    to_rem.add(target)
            else:
                if (
                    target + fast_pow(2, mm.n_qubits - op - 1) in counts
                    and target + fast_pow(2, mm.n_qubits - op - 1) not in t_list
                ):
                    cmplx_m = mm.fetch_m(measure_type=m_type.cmplx_hadamard, op_pos=op)
                    real_m = mm.fetch_m(measure_type=m_type.real_hadamard, op_pos=op)

                    target_arr[target] = infer_target(
                        target_idx=target,
                        source_idx=target + fast_pow(2, mm.n_qubits - op - 1),
                        source_val=target_arr[
                            target + fast_pow(2, mm.n_qubits - op - 1)
                        ],
                        h_measure=real_m,
                        v_measure=cmplx_m,
                    )
                    to_rem.add(target)

            t_list = t_list - to_rem

            if len(t_list) == 0:
                return

    # use MST to deal with infeasible ones
    graph = nx.complete_graph(counts)
    for a in graph.nodes():
        for b in graph.nodes():
            if a != b:
                graph[a][b]["weight"] = hamming(a, b)
    mst = nx.minimum_spanning_tree(graph)

    for n in t_list:
        # find best source
        edges = sorted(mst.edges(data=True), key=lambda node: node[2].get("weight", 1))
        cm_idx = 0
        mwe = edges[cm_idx]
        source = mwe[0] if mwe[1] == n else mwe[1]
        while source in t_list:
            mwe = edges[cm_idx]
            source = mwe[0] if mwe[1] == n else mwe[1]

        # construct measure operators with correct CNOT placement
        output = [int(x) for x in "{:0{size}b}".format(source ^ n, size=mm.n_qubits)]
        cnots = find_nonzero_positions(output)
        op_pos = cnots[0]
        cnots = cnots[1:]

        real_m = mm.fetch_cm(m_type.real_hadamard, cnots, op_pos)
        cmplx_m = mm.fetch_cm(m_type.cmplx_hadamard, cnots, op_pos)

        # infer target
        target_arr[n] = infer_target(
            target_idx=n ^ (1 << op_pos),  # bit shift to ensure correct structure
            source_idx=source,
            source_val=target_arr[source],
            h_measure=real_m,
            v_measure=cmplx_m,
        )


if __name__ == "__main__":
    print(" Done!")
    initial_states = [
        array([1/2, 1/sqrt(2), 1/sqrt(6), 1/sqrt(12)]),
        array([1/2, -1/sqrt(2), 1/sqrt(6), 1/sqrt(12)]), 
        array([1/2, 0, -2/sqrt(6), 1/sqrt(12)]),
        array([1/2, 0, 0, -3/sqrt(12)]),
        array([1/sqrt(2), -1/sqrt(2), 0, 0, 0, 0, 0, 0]),
        array([1/sqrt(6), 1/sqrt(6), -2/sqrt(6), 0, 0, 0, 0, 0])
    ]
    
    print()
    SHOTS = fast_pow(2, 14)
    print("Running inference at {} shots\n".format(SHOTS))
    
    for state in initial_states:
        r = pure_state_tomography(input_state=state, n_qubits=fast_log2(len(state)), precise=False, simulator=True, n_shots=SHOTS, verbose=True)
        print("Reconstructed vector:")
        print(r)
        print("% Error: {}\n".format(100 * linalg.norm(state - r)))

__author__ = "Kevin Wu and Shuhong Wang"
__credits__ = ["Kevin Wu", "Shuhong Wang"]
