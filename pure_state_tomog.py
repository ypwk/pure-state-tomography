#!/usr/bin/env python

print("Initializing . . .", end="", flush=True)

from numpy import ndarray, array, sqrt, asarray, zeros

from qiskit import QuantumCircuit, transpile, result, assemble
from qiskit import Aer, execute

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


def infer_target(
    target_idx, source_idx, source_val, n_qubits, h_measure, v_measure, op_pos
) -> ndarray:
    """Calculates and returns the value of an entry using previously inferred values in the measurement results.

    Args:
        target_idx (int): The index of the target value to infer
        source_idx (int): The index of the value to use to infer the target
        source_val (numpy.ndarray): The source value
        n_qubits (int): The number of qubits needed to represent the vector
        h_measure (numpy.ndarray): The array of measurements with the Hadamard gate
        v_measure (numpy.ndarray): The array of measurements with the alternate gate
        op_pos (int): Position of Hadamard/alternate operator in the tensor structure, zero indexed from the left

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
    qc.initialize(state, [0, 1])
    return qc


def run_circuit(qc, shots=1024) -> result.counts.Counts:
    """Runs the circuit on the simulator

    Args:
        qc (qiskit.QuantumCircuit): Quantum circuit to run
        shots (int): Number of shots to take

    Returns:
        numpy.ndarray: An array of result counts
    """
    aer_sim = Aer.get_backend("aer_simulator")
    t_qc = transpile(qc, aer_sim)
    qobj = assemble(t_qc, shots=shots)
    result = aer_sim.run(t_qc, shots=shots).result()
    counts = result.get_counts(qc)
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

    def __init__(self, m_state, n_qubits) -> None:
        self.n_qubits = n_qubits
        self.m_state = m_state
        self.num_measurements = 0
        self.__measurements = {
            m_type.identity: [None for _ in range(n_qubits)],
            m_type.cmplx_hadamard: [None for _ in range(n_qubits)],
            m_type.real_hadamard: [None for _ in range(n_qubits)],
        }
        pass

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

        # SHOTS = 2048
        # raw_result = run_circuit(circ_to_ex, SHOTS)

        # using statevector for precise execution
        simulator = Aer.get_backend("statevector_simulator")
        raw_result = execute(state_circuit, simulator).result()
        statevector = asarray(raw_result.get_statevector(state_circuit))

        res = zeros(fast_pow(2, self.m_state.num_qubits))
        for idx in range(len(statevector)):
            res[idx] = abs(
                statevector[idx].real * statevector[idx].real
                - statevector[idx].imag * statevector[idx].imag
            )

        self.num_measurements += 1
        self.__measurements[measure_type][op_pos] = res
        # print("Measurement for {} and {}: {}".format(measure_type, op_pos, res), flush=True)
        return res

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


def pure_state_tomography(input_state=None, n_qubits=2):
    """Uses the pure state tomography algorithm to infer the state of a hidden input_vector

    Args:
        input_state (numpy.ndarray, optional): The hidden input_vector to infer.
        n_qubits (int, optional): The number of qubits needed to represent the input_vector. Defaults to 2.
    """

    # TODO: Replace with meaningful input vector
    DIM = fast_pow(2, n_qubits)
    if input_state is None:  # generate random input vector if it is None
        # input_state = array([1 / 2, 1 / sqrt(2), 1 / sqrt(6), 1 / sqrt(12)])
        # input_state = array([1/2, -1/sqrt(2), 1/sqrt(6), 1/sqrt(12)])
        input_state = array([1 / 2, 0, -2 / sqrt(6), 1 / sqrt(12)])
        # input_state = array([1/2, 0, 0, -3/sqrt(12)])

    print("Input vector: {}".format(input_state), flush=True)

    mystery_state = create_circuit(input_state, n_qubits)

    # do initial identity measurement
    mm = meas_manager(m_state=mystery_state, n_qubits=n_qubits)
    
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
                for source in [target - fast_pow(2, mm.n_qubits - op - 1), target + fast_pow(2, mm.n_qubits - op - 1)]:
                    if source in counts and source not in t_list:
                        cmplx_m = mm.fetch_m(
                            measure_type=m_type.cmplx_hadamard, op_pos=op
                        )
                        real_m = mm.fetch_m(
                            measure_type=m_type.real_hadamard, op_pos=op
                        )

                        target_arr[target] = infer_target(
                            target_idx=target,
                            source_idx=source,
                            source_val=target_arr[source],
                            n_qubits=fast_log2(target_arr.shape[0]),
                            h_measure=real_m,
                            v_measure=cmplx_m,
                            op_pos=op,
                        )
                        to_rem.add(target)
            else:
                if target + fast_pow(2, mm.n_qubits - op - 1) in counts and target + fast_pow(2, mm.n_qubits - op - 1) not in t_list:
                    
                    cmplx_m = mm.fetch_m(
                        measure_type=m_type.cmplx_hadamard, op_pos=op
                    )
                    real_m = mm.fetch_m(
                        measure_type=m_type.real_hadamard, op_pos=op
                    )

                    target_arr[target] = infer_target(
                        target_idx=target,
                        source_idx=target + fast_pow(2, mm.n_qubits - op - 1),
                        source_val=target_arr[target + fast_pow(2, mm.n_qubits - op - 1)],
                        n_qubits=fast_log2(target_arr.shape[0]),
                        h_measure=real_m,
                        v_measure=cmplx_m,
                        op_pos=op,
                    )
                    to_rem.add(target)
            
            t_list = t_list - to_rem
            
            if len(t_list) == 0:
                return
    

if __name__ == "__main__":
    print(" Done!")
    r = pure_state_tomography(n_qubits=2)
    print("Reconstructed vector:")
    print(r)

__author__ = "Kevin Wu and Shuhong Wang"
__credits__ = ["Kevin Wu", "Shuhong Wang"]
