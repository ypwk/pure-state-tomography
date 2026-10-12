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
        verbose: bool = False,
        batch_size: int = 1,
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
        self.batch_size = batch_size
        self.n_qubits = mm.n_qubits
        self.partial_mixing = partial_mixing
        self.epsilon = epsilon
        DIM = putils.fast_pow(2, mm.n_qubits)
        state = np.zeros((self.batch_size, DIM, 2))

        self.verboseprint = print if verbose else (lambda *a, **k: None)

        mm.partial_mixing = partial_mixing

        if masked:
            self.identity_res = mm.add_measurement(
                qutils.m_type.identity, 0, clean=True
            )

        self.__iter_inf_helper(state, mm, dry=False)

        vector_form_result = np.array(
            [
                (state[b, :, 0] + 1j * state[b, :, 1])
                / np.linalg.norm((state[b, :, 0] + 1j * state[b, :, 1]))
                for b in range(batch_size)
            ]
        )

        # if masked:
        #     # shape (batch_size, dim), boolean
        #     mask = self.identity_res > 5e-2
        #     vector_form_result = vector_form_result * mask

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
        id_m = mm.fetch(qutils.m_type.identity, 0)
        if dry and type(id_m) is str:
            return
        nonzero_positions = qutils.find_nonzero_positions(id_m[0], epsilon=self.epsilon)

        if len(nonzero_positions) == 0:
            return

        pos = nonzero_positions[0]
        state[:, pos, 0] = np.sqrt(id_m[:, pos])
        t_list = set(nonzero_positions)
        t_list.remove(nonzero_positions[0])

        nonzero_positions = set(nonzero_positions)

        if dry:
            self.verboseprint("Dry run measurements:")
        else:
            self.verboseprint("Measurements:")

        # use MST
        graph = complete_graph(nonzero_positions)
        for a in graph.nodes():
            for b in graph.nodes():
                if a != b and (a in t_list or b in t_list):
                    graph[a][b]["weight"] = putils.hamming(a, b)
        mst = minimum_spanning_tree(graph)

        weighted_edges = [edge for edge in mst.edges(data=True) if "weight" in edge[2]]
        edges = sorted(weighted_edges, key=lambda node: node[2].get("weight", 1))

        edge_idx = 0
        while len(t_list) > 0 and edge_idx < len(edges):
            # find best source
            while edge_idx < len(edges):
                u, v, data = edges[edge_idx]
                if (u in t_list) ^ (v in t_list):
                    source_idx, target_idx = (u, v) if v in t_list else (v, u)
                    break
                edge_idx += 1
            else:
                break  # exhausted all edges

            # construct measure operators with correct CNOT placement
            difference_bitstring = [
                int(x)
                for x in "{:0{size}b}".format(source_idx ^ target_idx, size=mm.n_qubits)
            ]
            target_bitstring = [
                int(x) for x in "{:0{size}b}".format(target_idx, size=mm.n_qubits)
            ]
            # locations where source and target index bitstrings differ
            difference_nonzero_locations = qutils.find_nonzero_positions(
                difference_bitstring
            )
            # nonzero locations in target bitstring
            target_nonzero_locations = qutils.find_nonzero_positions(target_bitstring)
            op_pos = difference_nonzero_locations[0]
            nonzero = list(difference_nonzero_locations[1:])

            self.verboseprint(
                "Circuits for source index {} and target index {}:".format(
                    source_idx,
                    target_idx,
                )
            )

            if self.partial_mixing:
                hadamards = nonzero

                real_m = mm.fetch(
                    measure_type=qutils.m_type.real_hadamard,
                    hadamards=hadamards,
                    op_pos=op_pos,
                )
                cmplx_m = mm.fetch(
                    measure_type=qutils.m_type.cmplx_hadamard,
                    hadamards=hadamards,
                    op_pos=op_pos,
                )

                # print(f"Partial mixing real measurement: {real_m}")
                # print(f"Partial mixing cmplex measurement: {cmplx_m}")

                state[:, target_idx, :] = qutils.infer_partially_mixed_target(
                    target_idx=target_idx,
                    source_idx=source_idx,
                    hadamards=hadamards,
                    op_pos=op_pos,
                    source_val=state[:, source_idx, :],
                    h_measure=real_m,
                    v_measure=cmplx_m,
                )
            else:
                # figure out how to structure CNOTs
                cnots = []

                # find 1 position outside of nonzero
                for e in nonzero:
                    for t in target_nonzero_locations:
                        if t not in nonzero:
                            cnots.append([t, e])
                            break

                real_m = mm.fetch(
                    measure_type=qutils.m_type.real_hadamard, cnots=cnots, op_pos=op_pos
                )
                cmplx_m = mm.fetch(
                    measure_type=qutils.m_type.cmplx_hadamard,
                    cnots=cnots,
                    op_pos=op_pos,
                )

                # print(f"Standard mixing real measurement: {real_m}")
                # print(f"Standard mixing cmplex measurement: {cmplx_m}")

                corrected_target = target_idx
                for cnot in cnots:
                    corrected_target ^= 1 << (mm.n_qubits - 1 - cnot[1])

                state[:, target_idx, :] = qutils.infer_target(
                    target_idx=corrected_target,
                    source_idx=source_idx,
                    source_val=state[:, source_idx, :],
                    h_measure=real_m,
                    v_measure=cmplx_m,
                )

            self.verboseprint(
                f"Calculated target {target_idx} using source {source_idx}"
            )

            # print(f"real: {real_m}")
            # print(f"complex: {cmplx_m}")

            t_list.remove(target_idx)
            edge_idx += 1

        self.verboseprint("")
        if dry:
            mm.session.close()


__author__ = "Kevin Wu"

__credits__ = ["Kevin Wu"]
