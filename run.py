#!/usr/bin/env python3
import logging
import os
import re
import glob
import argparse
import yaml
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm

import numpy as np
from numpy import ndarray, reshape, linalg
import qiskit

import src.qutils as qutils
from src.measurements import measurement_manager
from src.algorithm import tomography

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("quantum_tomography")
# sacred artifacts under experiments/runs/sacred/
ex.observers.append(FileStorageObserver.create("experiments/runs"))


logger = logging.getLogger(__name__)  # module-level logger

logging.getLogger("qiskit").setLevel(logging.WARNING)


@ex.config
def cfg():
    # all text outputs from this run go here
    out_dir = "experiments/runs"

    # where to find per-experiment folders like 000_name, 012_process, ...
    experiment_config_root = "experiments/configs"

    # execution defaults
    execution = {
        "type": "simulator",   # "simulator" | "ibm_qpu"
        "n_shots": 2**14,
        "num_runs": 512,       # used only for simulator
        "verbosity": False,
    }

    # experiment-level defaults (each folder can override via config.yaml)
    experiment_defaults = {
        "epsilon": 5e-2,
        "tomography": "state",  # "state" | "process"
        "partial_mixing": False,      # None => auto rule below
        "job_file": None,
        "masked": True,
    }

    notes = ""  # optional note attached to sacred run

# ----------------- helpers -----------------


def _tomotype_from_str(s: str):
    return qutils.tomography_type.state if s == "state" else qutils.tomography_type.process


def _calc_fidelity(ideal: np.ndarray, actual: np.ndarray, tomotype) -> float:
    if tomotype is qutils.tomography_type.process:
        d = ideal.shape[0]
        inner = np.vdot(ideal.reshape(d*d), actual.reshape(d*d))
        return float(abs(inner) / d)
    return float(abs(np.vdot(ideal, actual)))


def _postprocess_log(state_obj: Union[ndarray, qiskit.QuantumCircuit],
                     res: Optional[np.ndarray], tomotype):
    if res is None:
        logger.info("No reconstruction returned.\n")
        return

    if isinstance(state_obj, np.ndarray):
        rec = (reshape(res, (state_obj.shape[0], state_obj.shape[0])).T
               if tomotype is qutils.tomography_type.process else res)
        logger.info("Reconstructed %s:\n%s",
                    "vector" if state_obj.ndim == 1 else "matrix", rec)
        logger.info("%% Error: %s\n", 100 * linalg.norm(state_obj - rec))
        return

    # QuantumCircuit
    if tomotype is qutils.tomography_type.process:
        ideal = qutils.circuit_to_unitary(state_obj)
        rec = reshape(res, (ideal.shape[0], ideal.shape[0])).T
    else:
        ideal = qutils.circuit_to_statevector(state_obj)
        rec = res

    logger.info("Original %s:\n%s",
                "vector" if ideal.ndim == 1 else "matrix", ideal)
    logger.info("Reconstructed %s:\n%s",
                "vector" if ideal.ndim == 1 else "matrix", rec)
    logger.info("Fidelity: %s\n", _calc_fidelity(ideal, rec, tomotype))


def _exp_dir_for_id(exp_root: str, exp_id: int) -> str:
    pattern = os.path.join(exp_root, f"{exp_id:03d}_*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No directory for experiment {exp_id} under {exp_root} (expected {pattern})")
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous experiment directory for {exp_id}: {matches}")
    return matches[0]


def _load_per_exp_cfg(exp_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(exp_dir, "config.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_job_file(job_file: Optional[str]) -> Optional[str]:
    if not job_file:
        return None
    # allow absolute or relative under ./jobs
    return job_file if os.path.isabs(job_file) else (os.path.join("jobs", job_file) if os.path.exists(os.path.join("jobs", job_file)) else None)


def _print_header(exp_id: int, exec_type, mm: measurement_manager):
    print(f"Experiment ID: {exp_id}")
    print(f"Backend: {exec_type}")
    print(f"Shots: {mm.n_shots}\n")

# ----------------- core single-exp runner -----------------


def _run_one(exp_id: int, out_dir: str, exp_root: str, execution: Dict[str, Any],
             exp_defaults: Dict[str, Any], talg: "tomography") -> None:
    exp_dir = _exp_dir_for_id(exp_root, exp_id)
    circ = qutils.load_from_experiment_dir(exp_dir)
    if circ is None:
        raise FileNotFoundError(
            f"{exp_dir}: missing circuit.qpy or circuit.qasm")

    local_cfg = _load_per_exp_cfg(exp_dir)
    cfg = {**exp_defaults, **local_cfg}

    epsilon = float(cfg.get("epsilon", 5e-2))
    tomotype = _tomotype_from_str(cfg.get("tomography", "state"))
    partial_mixing = cfg.get("partial_mixing", None)
    masked = bool(cfg.get("masked", True))
    job_file = _resolve_job_file(cfg.get("job_file"))

    if exp_id == 12:
        tomotype = qutils.tomography_type.process

    exec_type = qutils.execution_type.simulator if execution[
        "type"] == "simulator" else qutils.execution_type.ibm_qpu
    n_shots = int(execution["n_shots"])
    num_runs = int(execution["num_runs"])
    verbose = bool(execution["verbosity"])

    os.makedirs(out_dir, exist_ok=True)

    mm = measurement_manager(
        n_shots=n_shots, execution_type=exec_type, verbose=verbose)
    mm.set_state(tomography_type=tomotype, state=circ)
    _print_header(exp_id, exec_type, mm)

    for _ in tqdm(range(num_runs)):
        res = talg.pure_state_tomography(mm=mm, tomography_type=tomotype,
                                            verbose=verbose, job_file=job_file,
                                            partial_mixing=partial_mixing, epsilon=epsilon, masked=masked)
        _postprocess_log(circ, res, tomotype)


# ----------------- Sacred entrypoint -----------------


@ex.main
def main(out_dir, experiment_config_root, execution, experiment_defaults, notes, _run):   
    # find where Sacred is writing this run
    fs_observer = next(obs for obs in _run.observers if isinstance(obs, FileStorageObserver))
    run_dir = fs_observer.dir  # e.g. experiments/runs/1
    log_file = os.path.join(run_dir, "tomography.log")

    # Reset root logger to file only
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    ))
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info("Logging initialized for this run at %s", log_file)

    if notes:
        _run.info["notes"] = notes
    # the batch file path is passed via argv parsing below, then stashed in _run.info
    batch_ids: List[int] = _run.info.get("batch_ids", [])
    if not batch_ids:
        raise ValueError(
            "No experiment IDs were provided (batch file empty or missing).")
    talg = tomography()
    for exp_id in batch_ids:
        _run_one(exp_id, out_dir, experiment_config_root,
                 execution, experiment_defaults, talg)

# ----------------- CLI wrapper: single batch file -----------------


def _load_batch_ids(batch_file: str) -> List[int]:
    with open(batch_file, "r") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, list):
        ids = data
    else:
        ids = data.get("experiment_ids", [])
    # dedupe while preserving order
    seen = set()
    out = []
    for x in ids:
        i = int(x)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


if __name__ == "__main__":
    def parse_updates(kvs):
        """
        Parse dotted key=value pairs into a nested dict.
        Example:
          ["execution.num_runs=64", "experiment_defaults.epsilon=0.01", "out_dir=experiments/runs"]
        """
        def set_in(d, dotted, value):
            keys = dotted.split(".")
            cur = d
            for k in keys[:-1]:
                cur = cur.setdefault(k, {})
            # cast simple literals
            if isinstance(value, str):
                if re.fullmatch(r"[-+]?\d+", value):
                    value = int(value)
                elif re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", value, flags=re.I):
                    value = float(value)
                elif value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.lower() in ("null", "none"):
                    value = None
            cur[keys[-1]] = value

        out = {}
        for kv in kvs:
            if "=" not in kv:
                raise ValueError(f"--update expects key=value, got: {kv}")
            k, v = kv.split("=", 1)
            set_in(out, k.strip(), v.strip())
        return out

    ap = argparse.ArgumentParser(
        description="Run tomography from a single batch file of experiment IDs.")
    ap.add_argument("--batch-file", required=True,
                    help="Path like experiments/batches/<name>.yaml (bare list or {experiment_ids:[...]})")
    ap.add_argument("--update", nargs="*", default=[],
                    help="Optional dotted key=value overrides for Sacred config (e.g., execution.num_runs=64)")
    args = ap.parse_args()

    batch_ids = _load_batch_ids(args.batch_file)
    if not batch_ids:
        raise ValueError(f"{args.batch_file}: no experiment IDs found.")

    # minimal required config updates (your run dir + a helpful note)
    base_updates = {
        "out_dir": "experiments/runs",
        "notes": f"batch:{os.path.splitext(os.path.basename(args.batch_file))[0]}",
    }

    # merge user overrides (if any)
    user_updates = parse_updates(args.update)
    config_updates = {**base_updates, **user_updates}

    # hand the IDs to the Sacred run via `info`
    ex.run(
        config_updates=config_updates,
        info={"batch_ids": batch_ids},
        options={"--beat": False},  # optional; avoids double SIGINT handling
    )
