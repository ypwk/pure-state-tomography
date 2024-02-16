from numpy import (
    ndarray,
    reshape,
    linalg,
    pi,
    sqrt,
)
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

# read in configuration details
with open("config.ini", "r") as cf:
    cp = configparser.ConfigParser()
    cp.read_file(cf)
    api_token = cp.get("IBM", "token")
QiskitRuntimeService.save_account(
    channel="ibm_quantum", token=api_token, overwrite=True
)

talg = tomography()


def make_state(experiment_num: int):
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
    else:
        state = qiskit.QuantumCircuit(2)
        # state.x(0)
        # state.y(1)
        # state.cx(1, 0)
        state.h(0)
        state.u(pi / 4, 0, 0, 1)
        state.cx(0, 1)
    return state


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
                ).T * sqrt(state.shape[0])
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
                ).T * sqrt(state.shape[0])
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

            fprint("% Error: {}\n".format(100 * linalg.norm(state - res)))


epsilons = [
    5e-2,
    5e-3,
    5e-2,
    5e-2,
    5e-2,
    5e-2,
    5e-2,
    5e-5,
    5e-2,
    5e-5,
    5e-2,
    5e-3,
    5e-5,
]

execution_type = qutils.execution_type.simulator

experiment = sys.argv[1] if len(sys.argv) > 1 else 12
experiment = int(experiment)

VERBOSITY = False

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
    fprint(f"Index: {int(experiment)}")
    fprint(f"Hadamard: {int(experiment) % 2 == 1}")
    fprint(f"Experiment: {int(experiment) // 2}")
    fprint(f"Executing on: {execution_type}")
    fprint("Running inference at {} shots\n".format(mm.n_shots))

    run(
        mm=mm,
        tomography_type=(
            qutils.tomography_type.state
            if experiment < 12
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
        fprint(f"Index: {int(experiment)}")
        fprint(f"Hadamard: {int(experiment) % 2 == 1}")
        fprint(f"Experiment: {int(experiment) // 2}")
        fprint(f"Executing on: {execution_type}")
        fprint("Running inference at {} shots\n".format(mm.n_shots))

        for a in tqdm(range(512)):
            state = make_state(experiment)
            run(
                mm=mm,
                tomography_type=(
                    qutils.tomography_type.state
                    if experiment < 12
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
            experiment = exp

            fprint = putils.make_fprint(
                f"experiment_{experiment}_{execution_type.name}.txt"
            )
            fprint(f"Index: {int(experiment)}")
            fprint(f"Hadamard: {int(experiment) % 2 == 1}")
            fprint(f"Experiment: {int(experiment) // 2}")
            fprint(f"Executing on: {execution_type}")
            fprint("Running inference at {} shots\n".format(mm.n_shots))

            for a in tqdm(range(512)):
                state = make_state(experiment)
                run(
                    mm=mm,
                    tomography_type=(
                        qutils.tomography_type.state
                        if experiment < 12
                        else qutils.tomography_type.process
                    ),
                    experiment_num=experiment,
                    job_file=job_file,
                    state=state,
                    verbose=VERBOSITY,
                    hadamard=(experiment % 2 == 1),
                    epsilon=epsilons[experiment],
                )
