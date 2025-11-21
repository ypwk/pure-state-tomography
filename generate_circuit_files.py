#!/usr/bin/env python3
"""
Regenerate all original circuits (IDs 0–23) as OpenQASM 3 files and write configs.

Creates per experiment:
  experiments/configs/000_auto/
    - circuit.qasm
    - metadata.yaml
    - config.yaml            # per-experiment overrides (minimal)
    - standalone_run.yaml    # full Sacred config for single-experiment runs
"""

import os
import yaml
import qiskit
from qiskit.qasm3 import dumps as qasm3_dumps
from numpy import pi

# ---------- your original circuit factory ----------


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
        state.u(pi/4, 0, 0, 1)
        state.u(pi/4, 0, 0, 2)
        state.cx(2, 1)
        state.h(2)
    elif experiment_num < 10:
        state.u(pi/4, 0, 0, 1)
        state.u(pi/4, 0, 0, 2)
        state.cx(1, 2)
        state.h(1)
        state.cx(1, 0)
    elif experiment_num < 12:
        state.u(pi/4, 0, 0, 1)
        state.u(pi/4, 0, 0, 2)
        state.cx(2, 1)
        state.h(2)
        state.cx(1, 0)
        state.cx(2, 1)
    elif experiment_num < 14:  # process tomography
        state = qiskit.QuantumCircuit(2)
        state.ry(pi/3, 0)
        state.rx(pi/4, 1)
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
    elif experiment_num < 24:  # 6-qubit chain
        state = qiskit.QuantumCircuit(6)
        state.h(0)
        state.cx(0, 1)
        state.cx(1, 2)
        state.cx(2, 3)
        state.cx(3, 4)
        state.cx(4, 5)
    return state


# ---------- constants you might tweak ----------
OUT_ROOT = "experiments/configs"
RUNS_DIR = "experiments/runs"             # where your runner writes logs
EXP_ROOT_FOR_RUNNER = "experiments/configs"

# Your original epsilon table, index-aligned 0..23
EPSILONS = [
    5e-2, 5e-2,  # 0,1
    9e-2, 9e-2,  # 2,3
    9e-2, 9e-2,  # 4,5
    9e-2, 9e-2,  # 6,7
    9e-2, 9e-2,  # 8,9
    9e-2, 9e-2,  # 10,11
    9e-3, 9e-3,  # 12,13
    9e-2, 9e-2,  # 14,15
    9e-2, 9e-2,  # 16,17
    9e-2, 9e-2,  # 18,19
    9e-2, 9e-2,  # 20,21
    9e-2, 9e-2,  # 22,23
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_yaml(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def export_all():
    for exp_id in range(24):
        exp_dir = os.path.join(OUT_ROOT, f"{exp_id:03d}_auto")
        ensure_dir(exp_dir)

        circ = make_state(exp_id)
        # ---- circuit.qasm ----
        qasm_path = os.path.join(exp_dir, "circuit.qasm")
        with open(qasm_path, "w", encoding="utf-8") as f:
            f.write(qasm3_dumps(circ))

        # ---- metadata.yaml ----
        metadata = {
            "index": exp_id,
            "qregs": circ.num_qubits,
            "cregs": circ.num_clbits,
            "gates": [inst.operation.name for inst in circ.data],
        }
        write_yaml(os.path.join(exp_dir, "metadata.yaml"), metadata)

        # Determine per-exp params
        epsilon = float(EPSILONS[exp_id])
        tomography = "process" if exp_id == 12 or exp_id == 13 else "state"
        partial_mixing_default = exp_id % 2 == 1  # odd IDs get True, even get False

        # ---- config.yaml (minimal per-experiment overrides) ----
        per_exp_cfg = {
            "epsilon": epsilon,
            "tomography": tomography,
            "partial_mixing": partial_mixing_default,  # runner will default odd IDs to True
            "job_file": None,
            "masked": True,
            "notes": f"auto export id {exp_id:03d}",
        }
        write_yaml(os.path.join(exp_dir, "config.yaml"), per_exp_cfg)

        # ---- standalone_run.yaml ----
        standalone = {
            "out_dir": RUNS_DIR,
            "experiment_config_root": EXP_ROOT_FOR_RUNNER,
            "execution": {
                "type": "simulator",
                # "type": "statevector",
                "n_shots": 16384,
                "num_runs": 1,
                "verbosity": False,
            },
            "experiment_defaults": {  # used by our latest runner; optional
                "epsilon": 5e-2,
                "tomography": "state",
                "partial_mixing": None,
                "job_file": None,
                "masked": True,
            },
            # For a single-experiment run using the earlier Sacred style:
            "experiment": {
                "index": exp_id,
                "epsilon": epsilon,
                "tomography": tomography,
                "partial_mixing": partial_mixing_default,
                "job_file": None,
                "masked": True,
            },
            "notes": f"auto export id {exp_id:03d}",
        }
        write_yaml(os.path.join(exp_dir, "standalone_run.yaml"), standalone)

        print(
            f"[ok] exp {exp_id:03d} -> {qasm_path} (+ config.yaml, standalone_run.yaml)")


if __name__ == "__main__":
    export_all()
