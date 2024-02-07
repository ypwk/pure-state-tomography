#!/usr/bin/env python
"""
This module implements quantum state tomography using a pure state tomography algorithm.
It leverages various libraries such as Qiskit Aer for quantum simulation, NetworkX for
graph operations, and Numpy for numerical computations. The primary focus of this module
is to infer the state of a quantum system (represented as a vector) based on
measurements.

It defines one class:
- `tomography`: A class encapsulating the functionality for performing pure state
                tomography. It includes methods for setting up the quantum system,
                executing the tomography algorithm, and processing the results.

The `tomography` class provides a method `pure_state_tomography` to perform the state
tomography. It supports various configurations and options like the number of qubits,
usage of simulators, verbose output, and handling of Hadamard gates. Additionally,
it includes a private method `__iter_inf_helper` to assist in the iterative process of
inferring quantum states.


Example usage:
The module is intended to be used in quantum computing environments where state
tomography is required. Users should create an instance of the `tomography` class
and call its methods with the appropriate parameters to perform quantum state
tomography. See the experiment.ipynb file for more information.

Note:
This module assumes familiarity with quantum computing concepts and terminologies.
"""

from numpy import (
    array,
    ndarray,
    sqrt,
    zeros,
    linalg,
)

from networkx import complete_graph, minimum_spanning_tree
import putils
import qutils
from measurement_manager import measurement_manager


class tomography:
    def __init__(self) -> None:
        pass

    def pure_state_tomography(
        self,
        mm: measurement_manager,
        tomography_type: qutils.tomography_type,
        out_file: str,
        verbose: bool = False,
        job_file: str = None,
        hadamard: bool = False,
        epsilon: float = 5e-2,
    ):
        """
        Conducts pure state tomography on a quantum system to infer its state. This
        method uses measurements and quantum operations to reconstruct the state
        of a quantum system given a set of measurements.

        Args:
            mm (measurement_manager): An instance of the measurement manager that
                                    handles quantum measurements and operations.
            tomography_type (qutils.tomography_type): Type of tomography.
            out_file (str): File to output to.
            verbose (bool, optional): If set to True, the function will print detailed
                                    information about the tomography process. Defaults
                                    to False.
            job_file (str, optional): Path to a file containing precomputed jobs. If not
                                    provided, the function will compute and save jobs
                                    depending on the execution context. Defaults to
                                    None.
            hadamard (bool, optional): If set to True, applies a Hadamard transformation
                                    to the quantum system before and after tomography.
                                    Defaults to False.
            epsilon (float, optional): Epsilon value used to determine nonzero entries

        Returns:
            numpy.ndarray: A complex-valued array representing the inferred state of the
                        quantum system.

        Raises:
            ValueError: If the provided parameters are invalid or if an error occurs
                        during the tomography process.

        See experiment.ipynb for example usage.
        """
        self.fprint = putils.make_fprint(out_file)
        self.verbosefprint = self.fprint if verbose else lambda *a, **k: None
        self.verboseprint = print if verbose else lambda *a, **k: None

        self.n_qubits = mm.n_qubits
        self.hadamard = hadamard
        self.epsilon = epsilon
        DIM = putils.fast_pow(2, mm.n_qubits)
        res = zeros((DIM, 2))

        if (  # precise or noisy simulator
            mm.execution_type == qutils.execution_type.statevector
            or mm.execution_type == qutils.execution_type.simulator
        ):
            if hadamard:
                mm.apply_full_hadamard()
                self.identity_res = mm.add_clean_m(qutils.m_type.identity, 0)
            self.__iter_inf_helper(res, mm, dry=False)
        else:  # ibm qpu
            if hadamard:  # add hadamard
                mm.apply_full_hadamard()
                mm.dummy_measurement(qutils.m_type.identity, 0, clean=True)
            if job_file is None:
                mm.dummy_measurement(qutils.m_type.identity, 0)
                mm.to_job_file()
                self.verbosefprint(
                    "{} unitary operators applied. Need more!".format(
                        mm.num_measurements
                    ),
                )
                return
            else:
                measurement_count = mm.consume_job_file(job_file)
                if hadamard:
                    self.identity_res = mm.fetch_clean_m(qutils.m_type.identity, 0)
                else:
                    self.identity_res = mm.fetch_m(qutils.m_type.identity, 0)
                self.__iter_inf_helper(res, mm, dry=True)
                if len(mm) > measurement_count:
                    mm.to_job_file()
                    if len(mm) == measurement_count + mm.num_measurements:
                        self.verbosefprint(
                            "{} unitary operators applied. No more needed.".format(
                                mm.num_measurements
                            ),
                        )
                    else:
                        self.verbosefprint(
                            "{} unitary operators applied. Need more!".format(
                                mm.num_measurements
                            ),
                        )
                    return
                else:
                    res = zeros((DIM, 2))
                    self.__iter_inf_helper(res, mm, dry=False)

        self.verbosefprint(
            "Number of unitary operators applied: {}".format(mm.num_measurements),
        )

        vector_form_result = array([res[a][0] + 1j * res[a][1] for a in range(DIM)])
        if hadamard:
            self.verbosefprint(
                "Before Hadamard: {}".format(vector_form_result),
            )
            vector_form_result = putils.hadamard(vector_form_result)
            vector_form_result = [
                vector_form_result[i] if self.identity_res[i] > 5e-2 else 0
                for i in range(len(vector_form_result))
            ]
            self.verbosefprint(
                "After Hadamard: {}".format(vector_form_result),
            )

        if tomography_type is qutils.tomography_type.state:
            vector_form_result = vector_form_result / linalg.norm(vector_form_result)

        return vector_form_result

    def __iter_inf_helper(
        self,
        target_arr: ndarray,
        mm: measurement_manager,
        dry: bool,
    ) -> None:
        """An iterative implementation of the inference helper

        Args:
            target_arr (numpy.ndarray): The array with incomplete measurements
            mm (measurement_manager): Manager object keeping track of measurement
                                      values.
            dry (bool): Denotes whether or not this should be a dry run.
        """

        # do identity measurement to seed
        id_m = mm.fetch_m(qutils.m_type.identity, 0)
        if dry and type(id_m) is str:
            return
        counts = qutils.find_nonzero_positions(id_m, epsilon=self.epsilon)

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
                if a != b and (a in t_list or b in t_list):
                    graph[a][b]["weight"] = putils.hamming(a, b)
        mst = minimum_spanning_tree(graph)

        while len(t_list) > 0:
            # find best source
            weighted_edges = [
                edge for edge in mst.edges(data=True) if "weight" in edge[2]
            ]
            edges = sorted(weighted_edges, key=lambda node: node[2].get("weight", 1))

            cm_idx = 0
            minimum_weight_edge = edges[cm_idx]
            source = (
                minimum_weight_edge[0]
                if minimum_weight_edge[1] in t_list
                else minimum_weight_edge[1]
            )
            target = (
                minimum_weight_edge[1]
                if minimum_weight_edge[1] != source
                else minimum_weight_edge[0]
            )
            while source in t_list:
                minimum_weight_edge = edges[cm_idx]
                source = (
                    minimum_weight_edge[0]
                    if minimum_weight_edge[1] in t_list
                    else minimum_weight_edge[1]
                )
                target = (
                    minimum_weight_edge[1]
                    if minimum_weight_edge[1] != source
                    else minimum_weight_edge[0]
                )
                cm_idx += 1

            # construct measure operators with correct CNOT placement
            output = [
                int(x) for x in "{:0{size}b}".format(source ^ target, size=mm.n_qubits)
            ]
            target_nonzero = [
                int(x) for x in "{:0{size}b}".format(target, size=mm.n_qubits)
            ]
            nonzero = qutils.find_nonzero_positions(output)
            target_nonzero = qutils.find_nonzero_positions(target_nonzero)
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
                mm.dummy_measurement(qutils.m_type.real_hadamard, op_pos, cnots=cnots)
                mm.dummy_measurement(qutils.m_type.cmplx_hadamard, op_pos, cnots=cnots)
            else:
                if len(cnots) == 0:
                    real_m = mm.fetch_m(qutils.m_type.real_hadamard, op_pos)
                    cmplx_m = mm.fetch_m(qutils.m_type.cmplx_hadamard, op_pos)
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
            t_list.remove(target)

        self.verbosefprint("")
        if dry:
            mm.session.close()


__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
