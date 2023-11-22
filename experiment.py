from numpy import (
    ndarray,
    reshape,
    linalg,
    pi,
    sqrt,
)
from pure_state_tomography import tomography
import putils
import qutils
import qiskit
from measurement_manager import measurement_manager
import sys
import configparser
from qiskit_ibm_runtime import QiskitRuntimeService

with open("config.ini", "r") as cf:
    cp = configparser.ConfigParser()
    cp.read_file(cf)
    api_token = cp.get("IBM", "token")
QiskitRuntimeService.save_account(
    channel="ibm_quantum", token=api_token, overwrite=True
)

talg = tomography()


def run(
    mm: measurement_manager,
    tomography_type: qutils.tomography_type,
    state: ndarray | qiskit.QuantumCircuit,
    verbose: bool = True,
    job_file: str = None,
    hadamard: bool = False,
):
    putils.fprint("Running inference at {} shots\n".format(mm.n_shots))

    # reset measurement manager state
    mm.set_state(tomography_type=tomography_type, state=state)

    if type(state) is qiskit.QuantumCircuit:
        state = state.copy()

    # run pure state tomography
    res = talg.pure_state_tomography(
        mm=mm,
        tomography_type=tomography_type,
        verbose=verbose,
        job_file=job_file,
        hadamard=hadamard,
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
            putils.fprint(
                "Reconstructed {}:\n{}".format(
                    "vector" if state.ndim == 1 else "matrix", res
                )
            )
            putils.fprint("% Error: {}\n".format(100 * linalg.norm(state - res)))
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
            putils.fprint(
                "Original {}:\n{}".format(
                    "vector" if state.ndim == 1 else "matrix", state
                )
            )
            putils.fprint(
                "Reconstructed {}:\n{}".format(
                    "vector" if state.ndim == 1 else "matrix", res
                )
            )

            putils.fprint("% Error: {}\n".format(100 * linalg.norm(state - res)))


experiment = int(sys.argv[1])
VERBOSITY = False
mm = measurement_manager(
    n_shots=putils.fast_pow(2, 14),
    execution_type=qutils.execution_type.simulator,
    verbose=VERBOSITY,
)
for a in range(512):
    # put state code here
    state = qiskit.QuantumCircuit(3)
    if experiment < 2:
        state.h(0)
    elif experiment < 4:
        state.h(2)
        state.x(2)
        state.cnot(2, 1)
    elif experiment < 6:
        state.h(2)
        state.x(2)
        state.cnot(2, 1)
        state.cnot(2, 0)
    elif experiment < 8:
        state.u(pi / 4, 0, 0, 1)
        state.u(pi / 4, 0, 0, 2)
        state.cx(2, 1)
        state.h(2)
    elif experiment < 10:
        state.u(pi / 4, 0, 0, 1)
        state.u(pi / 4, 0, 0, 2)
        state.cx(2, 1)
        state.h(2)
        state.cx(2, 0)
    elif experiment < 12:
        state.u(pi / 4, 0, 0, 1)
        state.u(pi / 4, 0, 0, 2)
        state.cx(2, 1)
        state.h(2)
        state.cx(1, 0)
        state.cx(2, 1)
    else:
        state.h(0)
        state.h(1)
        state.h(2)
        state.x(0)
        state.u(pi / 4, 0, 0, 2)
    run(
        mm=mm,
        tomography_type=qutils.tomography_type.state,
        state=state,
        verbose=VERBOSITY,
        hadamard=(experiment % 2 == 1),
    )
