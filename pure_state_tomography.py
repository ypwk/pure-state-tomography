#!/usr/bin/env python

"""
This file provides utility and main execution code for quantum state tomography, aiming to infer the state of a hidden
    input vector.

Dependencies:
numpy: For array operations, linear algebra, and random number generation.
networkx: For graph structures and minimum spanning tree operations.
putils, qutils, mmng: Custom utility and manager modules.
Classes:
tomography: Contains methods for performing quantum state tomography, both in simulation and with actual quantum
    machines.

Methods:
pure_state_tomography: Uses the pure state tomography algorithm to deduce the state of a hidden input vector.
__iter_inf_helper: An iterative inference helper method.

Main Execution:
Demonstrates the application of the tomography class to infer predefined quantum states, logging the reconstructed
    vector and associated error.
Metadata:

Author: Kevin Wu
For comprehensive details on functions, methods, and their parameters, refer to individual docstrings within the code.
"""

from numpy import ndarray, array, sqrt, zeros, linalg, reshape

import putils
import qutils
import mmng
from networkx import complete_graph, minimum_spanning_tree


class tomography:
    def __init__(self) -> None:
        pass

    def pure_state_tomography(
        self,
        mm,
        n_qubits=2,
        simulator=True,
        verbose=False,
        job_file=None,
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
        self.verbosefprint = putils.fprint if verbose else lambda *a, **k: None
        self.verboseprint = print if verbose else lambda *a, **k: None

        self.n_qubits = mm.n_qubits
        DIM = putils.fast_pow(2, mm.n_qubits)
        res = zeros((DIM, 2))

        if simulator:
            self.__iter_inf_helper(res, mm, dry=False)
        else:
            if job_file is None:
                mm.dummy_measurement(qutils.m_type.identity, 0)
                mm.to_job_file()
            else:
                measurement_count = mm.consume_job_file(job_file)
                self.__iter_inf_helper(res, mm, dry=True)
                if len(mm) > measurement_count:
                    mm.to_job_file()
                else:
                    res = zeros((DIM, 2))
                    self.__iter_inf_helper(res, mm, dry=False)

        self.verbosefprint(
            "Number of unitary operators applied: {}".format(mm.num_measurements),
        )

        return [res[a][0] + 1j * res[a][1] for a in range(DIM)]

    def __iter_inf_helper(
        self,
        target_arr: ndarray,
        mm: mmng.meas_manager,
        dry: bool,
    ) -> None:
        """An iterative implementation of the inference helper

        Args:
            target_arr (numpy.ndarray): The array with incomplete measurements
            mm (meas_manager): Measure manager keeping track of measurement values.
            dry (bool): Denotes whether or not this should be a dry run.
        """

        # do identity measurement to seed
        id_m = mm.fetch_m(qutils.m_type.identity, 0)
        counts = qutils.find_nonzero_positions(id_m)

        if len(counts) == 0:
            return

        target_arr[counts[0]][0] = sqrt(id_m[counts[0]])
        t_list = set(counts)
        t_list.remove(counts[0])

        counts = set(counts)

        if dry:
            self.verbosefprint("Dry run measurements:")
        else:
            self.verbosefprint("Measurements:")

        for op in range(mm.n_qubits):
            to_rem = set()
            for target in t_list:
                if target & (1 << (mm.n_qubits - op - 1)):
                    if (
                        target - putils.fast_pow(2, mm.n_qubits - op - 1) in counts
                        and target - putils.fast_pow(2, mm.n_qubits - op - 1)
                        not in t_list
                    ):
                        self.verbosefprint(
                            "Circuits for source index {} and target index {}:".format(
                                target - putils.fast_pow(2, mm.n_qubits - op - 1),
                                target,
                            )
                        )
                        if dry:
                            mm.dummy_measurement(
                                measure_type=qutils.m_type.cmplx_hadamard, op_pos=op
                            )
                            mm.dummy_measurement(
                                measure_type=qutils.m_type.real_hadamard, op_pos=op
                            )
                        else:
                            cmplx_m = mm.fetch_m(
                                measure_type=qutils.m_type.cmplx_hadamard, op_pos=op
                            )
                            real_m = mm.fetch_m(
                                measure_type=qutils.m_type.real_hadamard, op_pos=op
                            )

                            target_arr[target] = qutils.infer_target(
                                target_idx=target,
                                source_idx=target
                                - putils.fast_pow(2, mm.n_qubits - op - 1),
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
                        and target + putils.fast_pow(2, mm.n_qubits - op - 1)
                        not in t_list
                    ):
                        if dry:
                            mm.dummy_measurement(
                                measure_type=qutils.m_type.cmplx_hadamard, op_pos=op
                            )
                            mm.dummy_measurement(
                                measure_type=qutils.m_type.real_hadamard, op_pos=op
                            )
                        else:
                            cmplx_m = mm.fetch_m(
                                measure_type=qutils.m_type.cmplx_hadamard, op_pos=op
                            )
                            real_m = mm.fetch_m(
                                measure_type=qutils.m_type.real_hadamard, op_pos=op
                            )

                            target_arr[target] = qutils.infer_target(
                                target_idx=target,
                                source_idx=target
                                + putils.fast_pow(2, mm.n_qubits - op - 1),
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
        graph = complete_graph(counts)
        for a in graph.nodes():
            for b in graph.nodes():
                if a != b:
                    graph[a][b]["weight"] = putils.hamming(a, b)
        mst = minimum_spanning_tree(graph)

        for target in t_list:
            # find best source
            edges = sorted(
                mst.edges(data=True), key=lambda node: node[2].get("weight", 1)
            )
            cm_idx = 0
            minimum_weight_edge = edges[cm_idx]
            source = (
                minimum_weight_edge[0]
                if minimum_weight_edge[1] == target
                else minimum_weight_edge[1]
            )
            while source in t_list:
                minimum_weight_edge = edges[cm_idx]
                source = (
                    minimum_weight_edge[0]
                    if minimum_weight_edge[1] == target
                    else minimum_weight_edge[1]
                )

            # construct measure operators with correct CNOT placement
            output = [
                int(x) for x in "{:0{size}b}".format(source ^ target, size=mm.n_qubits)
            ]
            target_nonzero = [
                int(x) for x in "{:0{size}b}".format(target, size=mm.n_qubits)
            ]
            nonzero = qutils.find_nonzero_positions(output)
            target_nonzero = qutils.find_nonzero_positions(output)
            op_pos = nonzero[0]
            nonzero = list(nonzero[1:])

            # figure out how to structure CNOTs
            cnots = []

            # find 1 position outside of nonzero
            for e in nonzero:
                for t in target_nonzero:
                    if t not in nonzero:
                        cnots.append([t, e])
                        break

            self.verbosefprint(
                "Circuits for source index {} and target index {}:".format(
                    source,
                    target,
                )
            )
            if dry:
                mm.dummy_measurement(qutils.m_type.real_hadamard, op_pos, cnots)
                mm.dummy_measurement(qutils.m_type.cmplx_hadamard, op_pos, cnots)
            else:
                real_m = mm.fetch_cm(qutils.m_type.real_hadamard, cnots, op_pos)
                cmplx_m = mm.fetch_cm(qutils.m_type.cmplx_hadamard, cnots, op_pos)

                corrected_target = target
                for cnot in cnots:
                    corrected_target ^= 1 << (mm.n_qubits - 1 - cnot[1])

                target_arr[target] = qutils.infer_target(
                    target_idx=corrected_target,
                    source_idx=source,
                    source_val=target_arr[source],
                    h_measure=real_m,
                    v_measure=cmplx_m,
                )

            mst.remove_edge(minimum_weight_edge[0], minimum_weight_edge[1])

        self.verbosefprint("")
        if dry:
            mm.session.close()


if __name__ == "__main__":
    initial_states = [
        # array([1 / 2, 1 / sqrt(2), 1 / sqrt(6), 1 / sqrt(12)]),
        # array([1 / 2, -1 / sqrt(2), 1 / sqrt(6), 1 / sqrt(12)]),
        # array([1 / 2, 0, -2 / sqrt(6), 1 / sqrt(12)]),
        # array([1 / 2, 0, 0, -3 / sqrt(12)]),
        array([1, 0, 0, 1]) / linalg.norm(array([1, 0, 0, 1])),
        array([0, 1, 1, 0]) / linalg.norm(array([1, 0, 0, 1])),
        # array([1 / sqrt(2) + 1j / sqrt(2), 0, 0, 1 / sqrt(2) + 1j / sqrt(2)]) / linalg.norm(array([1, 0, 0, 1])),
        # array([1, 0, 0, 1 / sqrt(2) + 1j / sqrt(2), 0, 0, 1, 0]) / linalg.norm(array([1, 0, 0, 1 / sqrt(2) + 1j / sqrt(2), 0, 0, 1, 0])),
        array(
            [
                [1 / 2, 1 / sqrt(2), 1 / sqrt(6), 1 / sqrt(12)],
                [1 / 2, -1 / sqrt(2), 1 / sqrt(6), 1 / sqrt(12)],
                [1 / 2, 0, -2 / sqrt(6), 1 / sqrt(12)],
                [1 / 2, 0, 0, -3 / sqrt(12)],
            ]
        ),
        # / linalg.norm(array([1, 0, 0, 1 / sqrt(2) + 1j / sqrt(2), 0, 0, 1, 0])),
        # array([1, 0, 0, 1, 0, 0, 1, 0]) / linalg.norm(array([1, 0, 0, 1, 0, 0, 1, 0])),
    ]
    job_names = [
        # "job_2023_10_09T_11_26_09.txt",
        # "job_2023_10_09T_11_26_17.txt",
        # "job_2023_10_09T_11_26_25.txt",
        # "job_2023_10_15T_15_45_57.txt",
    ]

    # job_names = [
    #     "job_2023_10_16T_01_45_47.txt",
    #     "job_2023_10_16T_01_45_54.txt",
    #     "job_2023_10_16T_01_45_59.txt",
    # ]
    VERBOSE = True
    SIMULATOR = True
    SHOTS = putils.fast_pow(2, 10)
    putils.fprint("Running inference at {} shots\n".format(SHOTS))

    talg = tomography()

    mm = mmng.meas_manager(
        n_shots=SHOTS,
        simulator=SIMULATOR,
        use_statevector=True,
        verbose=VERBOSE,
    )

    for state in range(len(initial_states)):
        mm.set_state(initial_states[state])
        res = talg.pure_state_tomography(
            mm=mm,
            n_qubits=putils.fast_log2(len(initial_states[state])),
            simulator=SIMULATOR,
            verbose=VERBOSE,
            job_file=job_names[state] if state < len(job_names) else None,
        )

        if initial_states[state].ndim > 1:
            res = reshape(
                res,
                (
                    initial_states[state].shape[0],
                    initial_states[state].shape[0],
                ),
            ).T * 2
        putils.fprint("Reconstructed vector:\n{}".format(res))
        putils.fprint(
            "% Error: {}\n".format(100 * linalg.norm(initial_states[state] - res))
        )

__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
