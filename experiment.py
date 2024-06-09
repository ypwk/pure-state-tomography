from numpy import (
    ndarray,
    reshape,
    linalg,
    pi,
    sqrt,
)
import numpy as np
from tqdm import tqdm
import configparser
import sys
import os

import qiskit
from qiskit_ibm_runtime import QiskitRuntimeService

import putils
import qutils
from measurement_manager import measurement_manager
from pure_state_tomography import tomography


def make_state(experiment_num: int):
    """
    Generates a quantum circuit based on the experiment number.

    Parameters:
        experiment_num (int): The experiment number.

    Returns:
        qiskit.QuantumCircuit: The generated quantum circuit.
    """
    state = qiskit.QuantumCircuit(3)
    if experiment_num < 2:
        state.h(0)
    elif experiment_num < 4:
        state.h(2)
        state.x(2)
        state.cx(2, 1)
    elif experiment_num < 6:
        state.h(2)
        state.x(2)
        state.cx(2, 1)
        state.cx(2, 0)
    elif experiment_num < 8:
        state.u(pi / 4, 0, 0, 1)
        state.u(pi / 4, 0, 0, 2)
        state.cx(2, 1)
        state.h(2)
    elif experiment_num < 10:
        state.u(pi / 4, 0, 0, 1)
        state.u(pi / 4, 0, 0, 2)
        state.cx(1, 2)
        state.h(1)
        state.cx(1, 0)
    elif experiment_num < 12:
        state.u(pi / 4, 0, 0, 1)
        state.u(pi / 4, 0, 0, 2)
        state.cx(2, 1)
        state.h(2)
        state.cx(1, 0)
        state.cx(2, 1)
    elif experiment_num < 14:  # process tomography
        state = qiskit.QuantumCircuit(2)
        # state.x(0)
        # state.y(1)
        # state.cx(1, 0)
        state.ry(pi / 3, 0)
        state.rx(pi / 4, 1)
        state.cx(0, 1)
    elif experiment_num < 16:  # 0000, 1111
        state = qiskit.QuantumCircuit(4)
        state.h(0)
        state.cx(0, 1)
        state.cx(1, 2)
        state.cx(2, 3)
    elif experiment_num < 18:  # 011 100
        state = qiskit.QuantumCircuit(3)
        state.h(0)
        state.x(2)
        state.cx(0, 1)
        state.cx(1, 2)
    elif experiment_num < 20:  # 00000, 11111
        state = qiskit.QuantumCircuit(5)
        state.h(0)
        state.cx(0, 1)
        state.cx(1, 2)
        state.cx(2, 3)
        state.cx(3, 4)
    elif experiment_num < 22:  # 110 001
        state = qiskit.QuantumCircuit(3)
        state.h(1)
        state.x(0)
        state.cx(1, 0)
        state.cx(1, 2)
    elif experiment_num < 24:  # 110 001
        state = qiskit.QuantumCircuit(6)
        state.h(0)
        state.cx(0, 1)
        state.cx(1, 2)
        state.cx(2, 3)
        state.cx(3, 4)
        state.cx(4, 5)
    return state


def calculate_fidelity(ideal, actual, type):
    """
    Calculates the fidelity between the ideal and actual quantum states.

    Parameters:
        ideal (ndarray): The ideal quantum state.
        actual (ndarray): The actual quantum state.
        type: type of fidelity to calculate, process or state

    Returns:
        float: The fidelity value.
    """
    if type == qutils.tomography_type.process:
        inp_dim = ideal.shape[0]
        ideal = np.reshape(ideal, (inp_dim * inp_dim, ))
        actual = np.reshape(actual, (inp_dim * inp_dim, )).T
        inner_product = np.vdot(ideal, actual)
        return np.abs(inner_product) / (inp_dim)
    elif type == qutils.tomography_type.state:
        inner_product = np.vdot(ideal, actual)
        return np.abs(inner_product)


def print_header(fprint, experiment, execution_type, mm):
    """
    Prints the header information for the experiment.

    Parameters:
        fprint (function): The function to print the information.
        experiment (int): The experiment number.
        execution_type: The type of execution (e.g., simulator or QPU).
        mm (measurement_manager): The measurement manager.
    """
    fprint(f"Index: {int(experiment)}")
    fprint(f"Hadamard: {int(experiment) % 2 == 1}")
    fprint(f"Experiment: {int(experiment) // 2}")
    fprint(f"Executing on: {execution_type}")
    fprint("Running inference at {} shots\n".format(mm.n_shots))


def run(
    mm: measurement_manager,
    tomography_type: qutils.tomography_type,
    state: ndarray | qiskit.QuantumCircuit,
    experiment_num: int,
    verbose: bool = True,
    job_file: str = None,
    hadamard: bool = False,
    epsilon: float = 5e-2,
):
    """
    Runs the quantum experiment and performs tomography.

    Parameters:
        mm (measurement_manager): The measurement manager.
        tomography_type (qutils.tomography_type): The type of tomography to perform.
        state (ndarray | qiskit.QuantumCircuit): The quantum state or circuit.
        experiment_num (int): The experiment number.
        verbose (bool): Whether to print verbose output. Defaults to True.
        job_file (str): The job file. Defaults to None.
        hadamard (bool): Whether to apply a Hadamard gate. Defaults to False.
        epsilon (float): The epsilon value for tomography. Defaults to 5e-2.
    """
    fprint = putils.make_fprint(
        f"experiment_{experiment_num}_{execution_type.name}.txt"
    )

    # reset measurement manager state
    mm.fprint = fprint
    mm.set_state(tomography_type=tomography_type, state=state)

    if type(state) is qiskit.QuantumCircuit:
        state = state.copy()

    # run pure state tomography
    res = talg.pure_state_tomography(
        mm=mm,
        tomography_type=tomography_type,
        out_file=f"experiment_{experiment_num}_{execution_type.name}.txt",
        verbose=verbose,
        job_file=job_file,
        hadamard=hadamard,
        epsilon=epsilon,
        masked=True,
    )

    if res is not None:
        if type(state) is ndarray:
            if tomography_type is qutils.tomography_type.process:
                res = reshape(
                    res,
                    (
                        state.shape[0],
                        state.shape[0],
                    ),
                ).T
            fprint(
                "Reconstructed {}:\n{}".format(
                    "vector" if state.ndim == 1 else "matrix", res
                )
            )
            fprint("% Error: {}\n".format(100 * linalg.norm(state - res)))
        elif type(state) is qiskit.QuantumCircuit:
            if tomography_type is qutils.tomography_type.process:
                state = qutils.circuit_to_unitary(state)
                res = reshape(
                    res,
                    (
                        state.shape[0],
                        state.shape[0],
                    ),
                ).T
            else:
                state = qutils.circuit_to_statevector(state)
            fprint(
                "Original {}:\n{}".format(
                    "vector" if state.ndim == 1 else "matrix", state
                )
            )
            fprint(
                "Reconstructed {}:\n{}".format(
                    "vector" if state.ndim == 1 else "matrix", res
                )
            )
            fprint(f"Fidelity: {calculate_fidelity(state, res, tomography_type)}")


# read in configuration details
with open("config.ini", "r") as cf:
    confp = configparser.ConfigParser()
    confp.read_file(cf)
    api_token = confp.get("IBM", "token")
QiskitRuntimeService.save_account(
    channel="ibm_quantum", token=api_token, overwrite=True
)


talg = tomography()

epsilons = [
    5e-2,  # 000, 001
    5e-2,
    5e-2,  # 000, 110
    5e-2,
    5e-2,  # 000, 111
    5e-2,
    5e-2,  # 000, 010, 100
    5e-5,
    5e-2,  # 000, 011, 100
    5e-5,
    5e-2,  # 000, 011, 110
    5e-3,
    5e-3,  # unitary
    5e-3,
    5e-2,  # 0000 1111
    5e-3,
    5e-2,  # 011 100
    5e-5,
    5e-2,  # 00000, 11111
    5e-5,
    5e-2,  # 110 011
    5e-5,
    5e-2,  # 000000, 111111
    5e-5,
]

execution_type = qutils.execution_type.simulator

experiment = int(sys.argv[1]) if len(sys.argv) > 1 else None

VERBOSITY = False

NUM_RUNS = 512

mm = measurement_manager(
    n_shots=putils.fast_pow(2, 14),
    execution_type=execution_type,
    out_file=f"experiment_{experiment}_{execution_type.name}.txt",
    verbose=VERBOSITY,
)

job_file = "experiment_{}.txt".format(experiment)
if not os.path.exists(os.path.join("jobs", job_file)):
    job_file = None


if execution_type is qutils.execution_type.ibm_qpu:
    state = make_state(experiment)
    fprint = putils.make_fprint(f"experiment_{experiment}_{execution_type.name}.txt")
    print_header(fprint, experiment, execution_type, mm)
    run(
        mm=mm,
        tomography_type=(
            qutils.tomography_type.state
            if experiment != 12
            else qutils.tomography_type.process
        ),
        experiment_num=experiment,
        job_file=job_file,
        state=state,
        verbose=VERBOSITY,
        hadamard=(experiment % 2 == 1),
        epsilon=epsilons[experiment],
    )
else:
    if experiment is not None:
        fprint = putils.make_fprint(
            f"experiment_{experiment}_{execution_type.name}.txt"
        )
        print_header(fprint, experiment, execution_type, mm)

        for a in tqdm(range(NUM_RUNS)):
            state = make_state(experiment)
            run(
                mm=mm,
                tomography_type=(
                    qutils.tomography_type.state
                    if experiment != 12
                    else qutils.tomography_type.process
                ),
                experiment_num=experiment,
                job_file=job_file,
                state=state,
                verbose=VERBOSITY,
                hadamard=(experiment % 2 == 1),
                epsilon=epsilons[experiment],
            )
    else:
        for exp in range(len(epsilons)):
            if exp == 13:
                continue

            experiment = exp

            fprint = putils.make_fprint(
                f"experiment_{experiment}_{execution_type.name}.txt"
            )
            print_header(fprint, experiment, execution_type, mm)

            for a in tqdm(range(NUM_RUNS)):
                state = make_state(experiment)
                run(
                    mm=mm,
                    tomography_type=(
                        qutils.tomography_type.state
                        if experiment != 12
                        else qutils.tomography_type.process
                    ),
                    experiment_num=experiment,
                    job_file=job_file,
                    state=state,
                    verbose=VERBOSITY,
                    hadamard=(experiment % 2 == 1),
                    epsilon=epsilons[experiment],
                )

__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
