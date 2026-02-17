#!/usr/bin/env python3
import logging
import os
import re
import glob
import argparse
import yaml
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy import ndarray, linalg
from scipy.linalg import polar
import qiskit
from qiskit.visualization import plot_error_map
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider
from time import perf_counter

import src.qutils as qutils
from src.measurements import measurement_manager
from src.algorithm import tomography
from src.noise_modeling import make_custom_noise_model

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("quantum_tomography")
# sacred artifacts under experiments/runs/sacred/
ex.observers.append(FileStorageObserver.create("experiments/runs"))


logger = logging.getLogger(__name__)  # module-level logger

logging.getLogger("qiskit").setLevel(logging.WARNING)


@ex.config
def cfg():
    out_dir = "experiments/runs"  # noqa: F841
    experiment_config_root = "experiments/configs"  # noqa: F841
    algorithm = {  # noqa: F841
        "name": "mst",   # e.g. "mst" | "gulbahar"
    }
    execution = {  # noqa: F841
        "type": "simulator",  # "simulator" | "ibm_qpu"
        "n_shots": 2**14,
        "num_runs": 32,
        "verbosity": False,
        "noise_model": {
            "mode": "custom",  # "fake_backend" | "custom" | "none"
            "fake_backend": "FakeHanoiV2",
            "custom_params": {
                "p_1q": 4.239e-4,
                "p_2q": 3.416e-3,
                "p_meas": 1e-2,
                "coherent_phase": 0,
                "n_qubits": 10,
            },
        },
    }

    notes = ""  # noqa: F841


# ----------------- helpers -----------------


def _tomotype_from_str(s: str):
    return (
        qutils.tomography_type.state if s == "state" else qutils.tomography_type.process
    )


def _calc_fidelity(ideal: np.ndarray, actual: np.ndarray, tomotype) -> float:
    if tomotype is qutils.tomography_type.process:
        d = ideal.shape[0]
        inner = np.vdot(ideal.reshape(d * d), actual.reshape(d * d))
        return float(abs(inner) / d)
    return float(abs(np.vdot(ideal, actual)))


def _postprocess_log(
    state_obj: Union[ndarray, qiskit.QuantumCircuit],
    res: Optional[np.ndarray],
    tomotype,
):
    if res is None:
        logger.info("No reconstruction returned.\n")
        return

    # detect batch dimension
    batched = res.ndim > 1 and res.shape[0] > 1
    batch_size = res.shape[0] if batched else 1
    # ---------------- case: ideal given as ndarray ----------------
    if isinstance(state_obj, np.ndarray):
        if tomotype is qutils.tomography_type.process:
            d = state_obj.shape[0]
            recs = (
                res.reshape(batch_size, d, d).transpose(0, 2, 1)
                if batched
                else res.reshape(d, d).T
            )
        else:
            recs = res

        if batched:
            errs = [100 * linalg.norm(state_obj - rec) for rec in recs]
            logger.info("Reconstructed %d batch results", batch_size)
            logger.info("Mean %% Error: %.3f, Std: %.3f",
                        np.mean(errs), np.std(errs))
        else:
            logger.info("Reconstructed:\n%s", recs)
            logger.info("%% Error: %s", 100 * linalg.norm(state_obj - recs))
        return

    # ---------------- case: ideal given as QuantumCircuit ----------------
    if tomotype is qutils.tomography_type.process:
        ideal = qutils.circuit_to_unitary(state_obj)
        d = ideal.shape[0]
        recs = (
            res.reshape(batch_size, d, d).transpose(0, 2, 1)
            if batched
            else res.reshape(d, d).T
        )

        # frobenius normalization
        if batched:
            recs = np.array([M / np.linalg.norm(M, "fro")
                            * np.sqrt(d) for M in recs])
        else:
            recs = recs / np.linalg.norm(recs, "fro") * np.sqrt(d)

        # polar decomposition to nearest unitary
        if batched:
            recs = np.stack([polar(mat)[0] for mat in recs], axis=0)
        else:
            recs, _ = polar(recs)
    else:
        ideal = qutils.circuit_to_statevector(state_obj)
        recs = res

    if batched:
        fids = [_calc_fidelity(ideal, rec, tomotype) for rec in recs]
        logger.info("Original:\n%s", ideal)
        logger.info("Processed %d reconstructions", batch_size)
        # for i, rec in enumerate(recs):
        #     logger.info("Reconstruction %d:\n%s", i, rec)
        logger.info("Mean Fidelity: %.4f, Std: %.4f",
                    np.mean(fids), np.std(fids))
        logger.info("Fidelities: %s", ", ".join(f"{f:.10f}" for f in fids))
    else:
        logger.info("Original:\n%s", ideal)
        logger.info("Reconstructed:\n%s", recs)
        logger.info("Fidelity: %.4f", _calc_fidelity(ideal, recs, tomotype))

    logger.info("\n")


def _exp_dir_for_id(exp_root: str, exp_id: int) -> str:
    pattern = os.path.join(exp_root, f"{exp_id:03d}_*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No directory for experiment {exp_id} under {exp_root} (expected {pattern})"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous experiment directory for {exp_id}: {matches}")
    return matches[0]


def _load_per_exp_cfg(exp_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(exp_dir, "standalone_run.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"{exp_dir}: missing standalone_run.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}


def _print_header(exp_id, execution, exp_config):
    logger.info(f"Experiment ID: {exp_id}")
    logger.info(f"Backend: {exp_config['execution']['type']}")
    if execution.get("type") == "simulator":
        if execution.get("noise_model"):
            logger.info(
                "Fake backend: %s",
                execution["noise_model"].get(
                    "fake_backend", "FakeTorino"
                ),
            )
            logger.info(
                "Noise mode: %s",
                execution["noise_model"].get("mode", "fake_backend"),
            )
        else:
            logger.info(
                "Fake backend: %s",
                execution.get("fake_backend", "FakeTorino"),
            )
    logger.info(f"Shots: {exp_config['execution']['n_shots']}")
    logger.info(
        f"Num runs (batch size): {exp_config['execution']['num_runs']}")
    logger.info(f"Tomography type: {exp_config['experiment']['tomography']}")
    logger.info(
        f"Partial mixing: {exp_config['experiment']['partial_mixing']}")
    logger.info(f"Epsilon: {exp_config['experiment']['epsilon']}")


def _get_fake_backend(name: str):
    backend_cls = getattr(fake_provider, name, None)
    if backend_cls is None or not callable(backend_cls):
        raise ValueError(
            f"Unknown fake backend '{name}' in execution.fake_backend")
    return backend_cls()


def _build_noise_model(execution_cfg: Dict[str, Any]) -> Optional[NoiseModel]:
    """Construct a noise model based on execution settings."""
    if execution_cfg.get("type") != "simulator":
        return None

    noise_cfg = execution_cfg.get("noise_model", {}) or {}
    mode = noise_cfg.get("mode", "fake_backend")

    if mode == "none":
        return None
    if mode == "custom":
        params = noise_cfg.get("custom_params", {}) or {}
        print("[noise] using custom noise model")
        return make_custom_noise_model(**params)
    if mode == "fake_backend":
        backend_name = noise_cfg.get("fake_backend") or execution_cfg.get(
            "fake_backend", "FakeTorino"
        )
        backend = _get_fake_backend(backend_name)

        # import matplotlib.pyplot as plt
        # plot_error_map(backend, figsize=(
        #     15, 12), show_title=True, qubit_coordinates=None)
        # plt.show()

        print(f"[noise] using fake backend {backend_name}")
        noise_model = NoiseModel.from_backend(
            backend,
            gate_error=True,
            readout_error=True,
            thermal_relaxation=True,
        )
        print("\n=== Noise Model Summary ===")
        print(noise_model)

        props = backend.properties()

        print("\n=== Backend Gate Error Summary ===")
        props = backend.properties()

        # One-qubit gates
        print("\nSingle-qubit gates and errors:")
        for g in props.gates:
            if len(g.qubits) == 1:
                q = g.qubits[0]
                ge = [p.value for p in g.parameters if p.name == 'gate_error']
                if ge:
                    print(f"  {g.gate:<3} on q{q}: error = {ge[0]:.3e}")

        # Two-qubit gates (CNOTs)
        print("\nTwo-qubit gates (CNOT) and errors:")
        for g in props.gates:
            if g.gate in ("cx", "cz"):
                qs = tuple(g.qubits)
                ge = [p.value for p in g.parameters if p.name == 'gate_error']
                if ge:
                    print(f"  {g.gate.upper()} {qs}: error = {ge[0]:.3e}")

        print("\n=================================\n")
        return noise_model

    raise ValueError(f"Unknown noise model mode: {mode}")


# ----------------- core single-exp runner -----------------


def _run_one(
    exp_id: int,
    out_dir: str,
    exp_root: str,
    algorithm: Dict[str, Any],
    execution: Dict[str, Any],
    talg: "tomography",
    noise_model: Optional[NoiseModel],
    _run=None,
) -> None:
    exp_dir = _exp_dir_for_id(exp_root, exp_id)
    circ = qutils.load_from_experiment_dir(exp_dir)
    if circ is None:
        raise FileNotFoundError(
            f"{exp_dir}: missing circuit.qpy or circuit.qasm")

    cfg = _load_per_exp_cfg(exp_dir)

    if _run is not None:
        if "experiment_configs" not in _run.info:
            _run.info["experiment_configs"] = {}
        _run.info["experiment_configs"][str(exp_id)] = cfg

    _print_header(exp_id, execution, cfg)

    epsilon = float(cfg["experiment"].get("epsilon", 5e-2))
    tomotype = _tomotype_from_str(cfg["experiment"].get("tomography", "state"))
    partial_mixing = cfg["experiment"].get("partial_mixing", None)
    masked = bool(cfg["execution"].get("masked", True))
    num_runs = int(cfg["execution"].get("num_runs", 512))
    n_shots = int(cfg["execution"].get("n_shots", 16384))
    verbose = bool(cfg["execution"].get("verbose", False))

    if exp_id == 12:
        tomotype = qutils.tomography_type.process

    if cfg["execution"]["type"] == "simulator":
        exec_type = qutils.execution_type.simulator
    elif cfg["execution"]["type"] == "statevector":
        exec_type = qutils.execution_type.statevector
    elif cfg["execution"]["type"] == "ibm_qpu":
        exec_type = qutils.execution_type.ibm_qpu
    else:
        raise ValueError(f"Unknown execution type: {execution['type']}")

    os.makedirs(out_dir, exist_ok=True)

    noise_cfg = execution.get("noise_model", {}) or {}

    if algorithm["name"] == "baseline":
        if execution["type"] == "simulator":
            backend_name = noise_cfg.get("fake_backend") or execution.get(
                "fake_backend", "FakeTorino"
            )
            backend = _get_fake_backend(backend_name)
            mm = measurement_manager(
                n_shots=n_shots,
                execution_type=exec_type,
                verbose=verbose,
                batch_size=num_runs,
                noise_model=noise_model,
            )
        else:
            mm = measurement_manager(
                n_shots=n_shots,
                execution_type=exec_type,
                verbose=verbose,
                batch_size=num_runs,
                noise_model=noise_model,
            )
        mm.set_state(tomography_type=tomotype, state=circ)

        res = talg.pure_state_tomography(
            mm=mm,
            tomography_type=tomotype,
            partial_mixing=partial_mixing,
            batch_size=num_runs,
            epsilon=epsilon,
            masked=masked,
        )
    else:
        res = gulbahar(
            state=circ
        )

    _postprocess_log(circ, res, tomotype)


# ----------------- Sacred entrypoint -----------------


@ex.main
def main(out_dir, experiment_config_root, algorithm, execution, notes, _run):
    # find where Sacred is writing this run
    fs_observer = next(
        obs for obs in _run.observers if isinstance(obs, FileStorageObserver)
    )
    run_dir = fs_observer.dir  # e.g. experiments/runs/1
    log_file = os.path.join(run_dir, "tomography.log")

    # Reset root logger to file only
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

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
            "No experiment IDs were provided (batch file empty or missing)."
        )
    talg = tomography()

    noise_cfg = execution.get("noise_model", {}) or {}
    noise_mode = noise_cfg.get("mode", "fake_backend")
    noise_model: Optional[NoiseModel] = _build_noise_model(execution)

    if execution["type"] == "simulator":
        if noise_mode == "fake_backend":
            backend_name = noise_cfg.get("fake_backend") or execution.get(
                "fake_backend", "FakeTorino"
            )
            logger.info(
                "Using noise model from fake backend: %s", backend_name)
        elif noise_mode == "custom":
            logger.info(
                "Using custom noise model params: %s",
                noise_cfg.get("custom_params", {}),
            )
        elif noise_mode == "none":
            logger.info("Noise model disabled for simulator.")

    total_start = perf_counter()

    for exp_id in batch_ids:
        exp_start = perf_counter()

        _run_one(
            exp_id, out_dir, experiment_config_root, algorithm, execution, talg, noise_model, _run
        )

        exp_end = perf_counter()
        exp_elapsed = exp_end - exp_start

        print(
            f"[TIMING] Experiment {exp_id} runtime: {exp_elapsed:.4f} seconds")

    # ------------------ Overall Timing End ------------------
    total_end = perf_counter()
    total_elapsed = total_end - total_start

    print("===================== TOTAL TIMING =====================")
    print(f"Total runtime (sec): {total_elapsed:.4f}")
    print(
        f"Average per experiment (sec): {total_elapsed / len(batch_ids):.4f}")
    print("=========================================================")


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
        description="Run tomography from a single batch file of experiment IDs."
    )
    ap.add_argument(
        "--batch-file",
        required=False,
        help="Path like experiments/batches/<name>.yaml (bare list or {experiment_ids:[...]})",
    )
    ap.add_argument(
        "--update",
        nargs="*",
        default=[],
        help="Optional dotted key=value overrides for Sacred config (e.g., execution.num_runs=64)",
    )
    args = ap.parse_args()

    if not args.batch_file:
        batches_dir = os.path.join("experiments", "batches")
        files = sorted(glob.glob(os.path.join(batches_dir, "*.yaml")))
        if not files:
            raise FileNotFoundError(f"No batch files found in {batches_dir}")

        print("Available batch files:")
        for i, f in enumerate(files):
            print(f"[{i}] {os.path.basename(f)}")

        while True:
            try:
                choice = int(input("Select a batch file by number: "))
                if 0 <= choice < len(files):
                    args.batch_file = files[choice]
                    break
                else:
                    print("Invalid selection, try again.")
            except ValueError:
                print("Please enter a valid number.")

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
