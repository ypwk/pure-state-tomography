#!/usr/bin/env python3
import argparse
import csv
import glob
import logging
import os
import re
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import yaml

import src.qutils as qutils
from src.algorithm import tomography
from src.measurements import measurement_manager

from sacred import Experiment
from sacred.observers import FileStorageObserver


ex = Experiment("admixture_sweep")
ex.observers.append(FileStorageObserver.create("experiments/runs_admixture_sweep"))

logger = logging.getLogger(__name__)
logging.getLogger("qiskit").setLevel(logging.WARNING)


@ex.config
def cfg():
    out_dir = "experiments/runs_admixture_sweep"  # noqa: F841
    experiment_config_root = "experiments/configs"  # noqa: F841
    execution = {  # noqa: F841
        "type": "simulator",
        "n_shots": 2**14,
        "num_runs": 32,
        "verbosity": False,
        "seed": 42,
    }
    admixture = {  # noqa: F841
        "mode": "maximally_mixed",
        "e_values": [0.0, 0.01, 0.02, 0.05, 0.1],
    }
    notes = ""  # noqa: F841


def _exp_dir_for_id(exp_root: str, exp_id: int) -> str:
    pattern = os.path.join(exp_root, f"{exp_id:03d}_*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No directory for experiment {exp_id} under {exp_root}."
        )
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous experiment directory for {exp_id}: {matches}")
    return matches[0]


def _load_per_exp_cfg(exp_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(exp_dir, "standalone_run.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"{exp_dir}: missing standalone_run.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_batch_ids(batch_file: str) -> List[int]:
    with open(batch_file, "r") as f:
        data = yaml.safe_load(f) or {}
    ids = data if isinstance(data, list) else data.get("experiment_ids", [])
    out = []
    seen = set()
    for x in ids:
        exp_id = int(x)
        if exp_id not in seen:
            seen.add(exp_id)
            out.append(exp_id)
    return out


def _execution_type(name: str):
    if name == "simulator":
        return qutils.execution_type.simulator
    if name == "statevector":
        return qutils.execution_type.statevector
    raise ValueError("Admixture sweep supports execution.type simulator or statevector.")


def _statevector_metrics(ideal: np.ndarray, reconstructed: np.ndarray) -> Dict[str, Any]:
    recs = reconstructed if reconstructed.ndim > 1 else reconstructed[None, :]
    fidelities = []
    mses = []

    for rec in recs:
        overlap = np.vdot(ideal, rec)
        fidelity = float(abs(overlap))
        if abs(overlap) > 0:
            rec = rec * np.exp(-1j * np.angle(overlap))
        mse = float(np.mean(np.abs(ideal - rec) ** 2))
        fidelities.append(fidelity)
        mses.append(mse)

    return {
        "mean_fidelity": float(np.mean(fidelities)),
        "std_fidelity": float(np.std(fidelities)),
        "mean_state_mse": float(np.mean(mses)),
        "std_state_mse": float(np.std(mses)),
        "fidelities": fidelities,
        "state_mses": mses,
    }


def _setup_file_logging(run_dir: str) -> None:
    log_file = os.path.join(run_dir, "tomography.log")
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)


@ex.main
def main(out_dir, experiment_config_root, execution, admixture, notes, _run):
    fs_observer = next(
        obs for obs in _run.observers if isinstance(obs, FileStorageObserver)
    )
    run_dir = fs_observer.dir
    _setup_file_logging(run_dir)

    if notes:
        _run.info["notes"] = notes

    batch_ids: List[int] = _run.info.get("batch_ids", [])
    if not batch_ids:
        raise ValueError("No experiment IDs were provided.")

    e_values = [float(e) for e in admixture.get("e_values", [])]
    if not e_values:
        raise ValueError("admixture.e_values must contain at least one value.")
    for e in e_values:
        if not 0.0 <= e <= 1.0:
            raise ValueError(f"admixture value out of range [0, 1]: {e}")

    os.makedirs(out_dir, exist_ok=True)
    talg = tomography()
    exec_type = _execution_type(execution["type"])
    rows: List[Dict[str, Any]] = []

    logger.info("Admixture model: rho_e = (1-e)|psi><psi| + e I/d")
    logger.info("Admixture e values: %s", e_values)
    logger.info("Batch IDs: %s", batch_ids)

    total_start = perf_counter()
    for e in e_values:
        logger.info("================ e = %.12g ================", e)
        for exp_id in batch_ids:
            exp_start = perf_counter()
            exp_dir = _exp_dir_for_id(experiment_config_root, exp_id)
            circ = qutils.load_from_experiment_dir(exp_dir)
            if circ is None:
                raise FileNotFoundError(f"{exp_dir}: missing circuit.qasm")
            cfg = _load_per_exp_cfg(exp_dir)

            tomotype = cfg["experiment"].get("tomography", "state")
            if tomotype != "state":
                raise ValueError(
                    f"Experiment {exp_id} uses tomography={tomotype}; "
                    "statevector MSE is only defined for state tomography."
                )

            partial_mixing = bool(cfg["experiment"].get("partial_mixing", False))
            epsilon = float(cfg["experiment"].get("epsilon", 5e-2))
            masked = bool(cfg["experiment"].get("masked", True))
            n_shots = int(execution.get("n_shots", cfg["execution"].get("n_shots", 16384)))
            num_runs = int(execution.get("num_runs", cfg["execution"].get("num_runs", 512)))
            verbose = bool(execution.get("verbosity", False))
            seed = execution.get("seed", 42)

            logger.info("Experiment ID: %s", exp_id)
            logger.info("Shots: %s", n_shots)
            logger.info("Num runs: %s", num_runs)
            logger.info("Partial mixing measurement scheme: %s", partial_mixing)
            logger.info("Epsilon threshold: %s", epsilon)

            mm = measurement_manager(
                n_shots=n_shots,
                execution_type=exec_type,
                verbose=verbose,
                batch_size=num_runs,
                admixture=e,
                admixture_mode=admixture.get("mode", "maximally_mixed"),
                admixture_sampling=True,
                seed=seed,
            )
            mm.set_state(tomography_type=qutils.tomography_type.state, state=circ)

            res = talg.pure_state_tomography(
                mm=mm,
                tomography_type=qutils.tomography_type.state,
                partial_mixing=partial_mixing,
                batch_size=num_runs,
                epsilon=epsilon,
                masked=masked,
            )
            ideal = qutils.circuit_to_statevector(circ)
            metrics = _statevector_metrics(ideal, res)
            elapsed = perf_counter() - exp_start

            logger.info(
                "Mean Fidelity: %.10f, Std: %.10f",
                metrics["mean_fidelity"],
                metrics["std_fidelity"],
            )
            logger.info(
                "Mean State MSE: %.10e, Std: %.10e",
                metrics["mean_state_mse"],
                metrics["std_state_mse"],
            )
            logger.info("Fidelities: %s", ", ".join(f"{x:.10f}" for x in metrics["fidelities"]))
            logger.info("State MSEs: %s", ", ".join(f"{x:.10e}" for x in metrics["state_mses"]))

            row = {
                "e": e,
                "experiment_id": exp_id,
                "partial_mixing": partial_mixing,
                "epsilon": epsilon,
                "n_shots": n_shots,
                "num_runs": num_runs,
                "elapsed_sec": elapsed,
                "mean_fidelity": metrics["mean_fidelity"],
                "std_fidelity": metrics["std_fidelity"],
                "mean_state_mse": metrics["mean_state_mse"],
                "std_state_mse": metrics["std_state_mse"],
            }
            rows.append(row)
            print(
                f"[admixture] e={e:g} exp={exp_id} "
                f"fid={metrics['mean_fidelity']:.6f} "
                f"mse={metrics['mean_state_mse']:.6e}"
            )

    _run.info["admixture_results"] = rows
    out_csv = os.path.join(run_dir, "admixture_sweep.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _run.add_artifact(out_csv)

    total_elapsed = perf_counter() - total_start
    logger.info("Total runtime: %.4f seconds", total_elapsed)
    print(f"[admixture] results written to {out_csv}")


def _parse_updates(kvs):
    def set_in(d, dotted, value):
        keys = dotted.split(".")
        cur = d
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        if isinstance(value, str):
            if re.fullmatch(r"[-+]?\d+", value):
                value = int(value)
            elif re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", value, flags=re.I):
                value = float(value)
            elif value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.lower() in ("null", "none"):
                value = None
            elif "," in value:
                value = [float(v.strip()) for v in value.split(",") if v.strip()]
        cur[keys[-1]] = value

    out = {}
    for kv in kvs:
        if "=" not in kv:
            raise ValueError(f"--update expects key=value, got: {kv}")
        key, value = kv.split("=", 1)
        set_in(out, key.strip(), value.strip())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a maximally mixed admixture sweep for state tomography."
    )
    parser.add_argument("--batch-file", required=True)
    parser.add_argument(
        "--update",
        nargs="*",
        default=[],
        help=(
            "Dotted Sacred config overrides. Example: "
            "admixture.e_values=0,0.01,0.05 execution.num_runs=64"
        ),
    )
    args = parser.parse_args()

    batch_ids = _load_batch_ids(args.batch_file)
    if not batch_ids:
        raise ValueError(f"{args.batch_file}: no experiment IDs found.")

    config_updates = _parse_updates(args.update)
    ex.run(
        config_updates=config_updates,
        info={"batch_ids": batch_ids},
        options={"--beat": False},
    )
