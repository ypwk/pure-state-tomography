#!/usr/bin/env python

from numpy import ndarray, array, sqrt, zeros, linalg
import numpy.random as nprandom

import putils
import qutils
import mmng
import networkx as nx


def pure_state_tomography(
    input_state=None,
    n_qubits=2,
    precise=False,
    simulator=True,
    n_shots=8192,
    verbose=False,
):
    """Uses the pure state tomography algorithm to infer the state of a hidden input_vector

    Args:
        input_state (numpy.ndarray, optional): The hidden input_vector to infer.
        n_qubits (int, optional): The number of qubits needed to represent the input_vector. Defaults to 2.
        precise (bool): Whether or not to use precise measurements on Qiskit
        simulator (bool): Whether or not to use Qiskit simulator or IBM quantum machine
        n_shots (int): The number of shots to use for imprecise measurements on Qiskit. Only comes into effect when
            precise is False.
        verbose (bool): Whether to print detailed information
    """

    DIM = putils.fast_pow(2, n_qubits)
    if input_state is None:  # generate random input vector if it is None
        input_state = nprandom.rand(DIM) + 1j * nprandom.rand(DIM)
        input_state = input_state / linalg.norm(input_state)

    putils.fprint("Input vector: {}".format(input_state), flush=True, filename='output.txt')

    mystery_state = qutils.create_circuit(input_state, n_qubits)

    # do initial identity measurement
    mm = mmng.meas_manager(
        m_state=mystery_state,
        n_qubits=n_qubits,
        n_shots=n_shots,
        simulator=simulator,
        use_statevector=precise,
        verbose=verbose,
    )

    res = zeros((DIM, 2))
    __iter_inf_helper(res, mm)

    putils.fprint("Number of unitary operators applied: {}".format(mm.num_measurements), filename='output.txt')

    return [res[a][0] + 1j * res[a][1] for a in range(DIM)]


def __iter_inf_helper(
    target_arr: ndarray,
    mm: mmng.meas_manager,
) -> None:
    """An iterative implementation of the inference helper

    Args:
        target_arr (numpy.ndarray): The array with incomplete measurements
        mm (meas_manager): Measure manager keeping track of measurement values.
    """

    # do identity measurement to seed
    id_m = mm.add_m(qutils.m_type.identity, 0)
    counts = qutils.find_nonzero_positions(id_m)

    target_arr[counts[0]][0] = sqrt(id_m[0])
    t_list = set(counts)
    t_list.remove(counts[0])

    counts = set(counts)

    for op in range(mm.n_qubits):
        to_rem = set()
        for target in t_list:
            if target & (1 << (mm.n_qubits - op - 1)):
                if (
                    target - putils.fast_pow(2, mm.n_qubits - op - 1) in counts
                    and target - putils.fast_pow(2, mm.n_qubits - op - 1) not in t_list
                ):
                    cmplx_m = mm.fetch_m(measure_type=qutils.m_type.cmplx_hadamard, op_pos=op)
                    real_m = mm.fetch_m(measure_type=qutils.m_type.real_hadamard, op_pos=op)

                    target_arr[target] = qutils.infer_target(
                        target_idx=target,
                        source_idx=target - putils.fast_pow(2, mm.n_qubits - op - 1),
                        source_val=target_arr[
                            target - putils.fast_pow(2, mm.n_qubits - op - 1)
                        ],
                        h_measure=real_m,
                        v_measure=cmplx_m,
                    )
                    to_rem.add(target)
            else:
                if (
                    target + putils.fast_pow(2, mm.n_qubits - op - 1) in counts
                    and target + putils.fast_pow(2, mm.n_qubits - op - 1) not in t_list
                ):
                    cmplx_m = mm.fetch_m(measure_type=qutils.m_type.cmplx_hadamard, op_pos=op)
                    real_m = mm.fetch_m(measure_type=qutils.m_type.real_hadamard, op_pos=op)

                    target_arr[target] = qutils.infer_target(
                        target_idx=target,
                        source_idx=target + putils.fast_pow(2, mm.n_qubits - op - 1),
                        source_val=target_arr[
                            target + putils.fast_pow(2, mm.n_qubits - op - 1)
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
                graph[a][b]["weight"] = putils.hamming(a, b)
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
        cnots = qutils.find_nonzero_positions(output)
        op_pos = cnots[0]
        cnots = cnots[1:]

        real_m = mm.fetch_cm(qutils.m_type.real_hadamard, cnots, op_pos)
        cmplx_m = mm.fetch_cm(qutils.m_type.cmplx_hadamard, cnots, op_pos)

        # infer target
        target_arr[n] = qutils.infer_target(
            target_idx=n ^ (1 << op_pos),  # bit shift to ensure correct structure
            source_idx=source,
            source_val=target_arr[source],
            h_measure=real_m,
            v_measure=cmplx_m,
        )


if __name__ == "__main__":
    initial_states = [
        array([1 / 2, 1 / sqrt(2), 1 / sqrt(6), 1 / sqrt(12)]),
        array([1 / 2, -1 / sqrt(2), 1 / sqrt(6), 1 / sqrt(12)]),
        array([1 / 2, 0, -2 / sqrt(6), 1 / sqrt(12)]),
        array([1 / 2, 0, 0, -3 / sqrt(12)]),
        array([1 / sqrt(2), -1 / sqrt(2), 0, 0, 0, 0, 0, 0]),
        array([1 / sqrt(6), 1 / sqrt(6), -2 / sqrt(6), 0, 0, 0, 0, 0]),
    ]

    SHOTS = putils.fast_pow(2, 10)
    putils.fprint("Running inference at {} shots\n".format(SHOTS), filename='output.txt')

    for state in initial_states:
        res = pure_state_tomography(
            input_state=state,
            n_qubits=putils.fast_log2(len(state)),
            precise=True,
            simulator=True,
            n_shots=SHOTS,
            verbose=True,
        )
        putils.fprint("Reconstructed vector:\n{}".format(res), filename='output.txt')
        putils.fprint("% Error: {}\n".format(100 * linalg.norm(state - res)), filename='output.txt')


__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu", "Shuhong Wang"]
