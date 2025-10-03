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

import numpy as np
from networkx import complete_graph, minimum_spanning_tree
import src.putils as putils
import src.qutils as qutils
from src.measurements import measurement_manager


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
        partial_mixing: bool = True,
        epsilon: float = 5e-2,
        masked: bool = True,
    ) -> np.ndarray | None:
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
            partial_mixing (bool, optional): If set to True, applies a Hadamard transformation
                                             to the quantum system before and after tomography
                                             instead of CNOTs. Defaults to True.
            epsilon (float, optional): Epsilon value used to determine nonzero entries
            masked (bool, optional)

        Returns:
            numpy.np.ndarray: A complex-valued np.array representing the inferred state of the
                        quantum system.

        Raises:
            ValueError: If the provided parameters are invalid or if an error occurs
                        during the tomography process.

        See experiment.ipynb for example usage.
        """
        self.n_qubits = mm.n_qubits
        self.partial_mixing = partial_mixing
        self.epsilon = epsilon
        DIM = putils.fast_pow(2, mm.n_qubits)
        state = np.zeros((DIM, 2))
        
        mm.partial_mixing = partial_mixing

        if masked:
            self.identity_res = mm.add_clean_m(qutils.m_type.identity, 0)

        self.__iter_inf_helper(state, mm, dry=False)

        vector_form_result = np.array(
            [state[a][0] + 1j * state[a][1] for a in range(DIM)])

        if masked:
            vector_form_result = [
                vector_form_result[i] if self.identity_res[i] > 5e-2 else 0
                for i in range(len(vector_form_result))
            ]

        if tomography_type is qutils.tomography_type.state:
            vector_form_result = vector_form_result / \
                np.linalg.norm(vector_form_result)
        elif tomography_type is qutils.tomography_type.process:
            vector_form_result = [
                _ * np.sqrt(np.sqrt(len(vector_form_result))) for _ in vector_form_result]

        return vector_form_result

    def __iter_inf_helper(
        self,
        state: np.ndarray,
        mm: measurement_manager,
        dry: bool,
    ) -> None:
        """An iterative implementation of the inference helper

        Args:
            state (numpy.np.ndarray): The np.array with incomplete measurements
            mm (measurement_manager): Manager object keeping track of measurement
                                      values.
            dry (bool): Denotes whether or not this should be a dry run.
        """

        # do identity measurement to seed
        id_m = mm.fetch_m(qutils.m_type.identity, 0)
        if dry and type(id_m) is str:
            return
        nonzero_positions = qutils.find_nonzero_positions(
            id_m, epsilon=self.epsilon)

        if len(nonzero_positions) == 0:
            return

        state[nonzero_positions[0]][0] = np.sqrt(
            id_m[nonzero_positions[0]])
        t_list = set(nonzero_positions)
        t_list.remove(nonzero_positions[0])

        nonzero_positions = set(nonzero_positions)

        if dry:
            self.verbosefprint("Dry run measurements:")
        else:
            self.verbosefprint("Measurements:")

        # use MST
        graph = complete_graph(nonzero_positions)
        for a in graph.nodes():
            for b in graph.nodes():
                if a != b and (a in t_list or b in t_list):
                    graph[a][b]["weight"] = putils.hamming(a, b)
        mst = minimum_spanning_tree(graph)

        weighted_edges = [
            edge for edge in mst.edges(data=True) if "weight" in edge[2]
        ]
        edges = sorted(
            weighted_edges, key=lambda node: node[2].get("weight", 1))

        edge_idx = 0
        while len(t_list) > 0 and edge_idx < len(edges):
            # find best source
            while edge_idx < len(edges):
                u, v, data = edges[edge_idx]
                if (u in t_list) ^ (v in t_list):
                    source, target = (u, v) if v in t_list else (v, u)
                    break
                edge_idx += 1
            else:
                break  # exhausted all edges

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
                mm.dummy_measurement(
                    qutils.m_type.real_hadamard, op_pos, cnots=cnots)
                mm.dummy_measurement(
                    qutils.m_type.cmplx_hadamard, op_pos, cnots=cnots)
            else:
                real_m = mm.fetch(
                    measure_type=qutils.m_type.real_hadamard, cnots=cnots, op_pos=op_pos)
                cmplx_m = mm.fetch(
                    measure_type=qutils.m_type.cmplx_hadamard, cnots=cnots, op_pos=op_pos)

                corrected_target = target
                for cnot in cnots:
                    corrected_target ^= 1 << (mm.n_qubits - 1 - cnot[1])

                state[target] = qutils.infer_target(
                    target_idx=corrected_target,
                    source_idx=source,
                    source_val=state[source],
                    h_measure=real_m,
                    v_measure=cmplx_m,
                )

                self.verbosefprint(
                    f"Calculated target {corrected_target} using source {source}")

            t_list.remove(target)
            edge_idx += 1

        self.verbosefprint("")
        if dry:
            mm.session.close()

    def __iter_inf_partial_mixed(
        self,
        state: np.ndarray,
        start_idx: int, 
        end_idx: int,
        mm: measurement_manager,
        dry: bool,
    ) -> None:
        """An iterative implementation of the inference helper, using Hadamard gates
        instead of CNOTs.

        Args:
            state (numpy.np.ndarray): The np.array with incomplete measurements
            start_idx (int): The starting index of the range to process.
            end_idx (int): The ending index of the range to process.
            mm (measurement_manager): Manager object keeping track of measurement
                                      values.
            dry (bool): Denotes whether or not this should be a dry run.
        """

        # take measurements of both 

__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
