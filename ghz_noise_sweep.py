#!/usr/bin/env python3
import logging
import os
from typing import Optional
import matplotlib.pyplot as plt

import numpy as np
import qiskit
from qiskit_aer.noise import NoiseModel

from sacred import Experiment
from sacred.observers import FileStorageObserver

# your modules
import src.qutils as qutils
from src.measurements import measurement_manager
from src.algorithm import tomography
from src.noise_modeling import make_custom_noise_model


# ============================================================
# === Sacred Setup ==========================================
# ============================================================

ex = Experiment("ghz_noise_sweep")
ex.observers.append(FileStorageObserver.create("experiments/runs_noise_sweep"))

logger = logging.getLogger(__name__)
logging.getLogger("qiskit").setLevel(logging.WARNING)


# ============================================================
# === Default Config =========================================
# ============================================================

@ex.config
def cfg():
    out_dir = "experiments/runs_noise_sweep"

    # size of GHZ state
    ghz_size = 6

    # tomography settings
    n_shots = 2**14
    num_runs = 64
    epsilon = 9e-2

    # noise sweep ranges
    sweep = {
        "p_1q": [4.239e-4],
        "p_2q": [10**x for x in np.linspace(-4, -2, 10)],
        "p_meas": [10**x for x in np.linspace(-3, -1.3, 10)],
        "coherent_phase": [0.0],
    }

    notes = ""


# ============================================================
# === Fidelity Helper ========================================
# ============================================================

def compute_fidelity(ideal: np.ndarray, actual: np.ndarray) -> float:
    return float(abs(np.vdot(ideal, actual)))


# ============================================================
# === Single Tomography Run (PM or ENT) ======================
# ============================================================

def run_single_tomo(
    qc: qiskit.QuantumCircuit,
    n_shots: int,
    num_runs: int,
    epsilon: float,
    noise_model: Optional[NoiseModel],
    partial_mixing: bool,
):

    talg = tomography()

    mm = measurement_manager(
        n_shots=n_shots,
        execution_type=qutils.execution_type.simulator,
        verbose=False,
        batch_size=num_runs,
        noise_model=noise_model,
    )
    mm.set_state(tomography_type=qutils.tomography_type.state, state=qc)

    res = talg.pure_state_tomography(
        mm=mm,
        tomography_type=qutils.tomography_type.state,
        partial_mixing=partial_mixing,
        batch_size=num_runs,
        epsilon=epsilon,
        masked=True,
    )

    ideal = qutils.circuit_to_statevector(qc)

    if res.ndim == 1:
        return compute_fidelity(ideal, res)

    fids = [compute_fidelity(ideal, r) for r in res]
    return float(np.mean(fids))


# ============================================================
# === Heatmap Plotting =======================================
# ============================================================

def plot_dual_heatmaps(results, sweep, out_dir, ghz_size):

    p2_vals = sorted(sweep["p_2q"])
    pm_vals = sorted(sweep["p_meas"])

    ENT = np.zeros((len(pm_vals), len(p2_vals)))
    PM = np.zeros((len(pm_vals), len(p2_vals)))

    for r in results:
        p2 = r["p_2q"]
        pm = r["p_meas"]
        i = pm_vals.index(pm)
        j = p2_vals.index(p2)

        ENT[i, j] = r["ENT_fidelity"]
        PM[i, j]  = r["PM_fidelity"]

    DIFF = PM - ENT
    RATIO = PM / ENT

    def _plot_matrix(M, title, fname, cmap="viridis", vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(7, 5))

        im = ax.imshow(
            M,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[
                np.log10(p2_vals[0]), np.log10(p2_vals[-1]),
                np.log10(pm_vals[0]), np.log10(pm_vals[-1])
            ],
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel("log10(p_2q)")
        ax.set_ylabel("log10(p_meas)")
        ax.set_title(title)

        ax.set_xticks(np.log10(p2_vals))
        ax.set_xticklabels([f"{v:.0e}" for v in p2_vals])

        ax.set_yticks(np.log10(pm_vals))
        ax.set_yticklabels([f"{v:.0e}" for v in pm_vals])

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Value")

        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[heatmap] Saved {title}: {out_path}")

    _plot_matrix(
        ENT,
        f"ENT Fidelity (GHZ-{ghz_size})",
        f"heatmap_ENT_ghz{ghz_size}.png",
        cmap="viridis",
        vmin=ENT.min(),
        vmax=1.0,
    )

    _plot_matrix(
        PM,
        f"PM Fidelity (GHZ-{ghz_size})",
        f"heatmap_PM_ghz{ghz_size}.png",
        cmap="viridis",
        vmin=PM.min(),
        vmax=1.0,
    )

    _plot_matrix(
        DIFF,
        f"Difference (PM − ENT)",
        f"heatmap_DIFF_ghz{ghz_size}.png",
        cmap="coolwarm",
        vmin=-np.max(np.abs(DIFF)),
        vmax=np.max(np.abs(DIFF)),
    )

    _plot_matrix(
        RATIO,
        f"Ratio (PM / ENT)",
        f"heatmap_RATIO_ghz{ghz_size}.png",
        cmap="plasma",
    )


# ============================================================
# === GHZ Circuit Builder ====================================
# ============================================================

def make_ghz(n: int) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


# ============================================================
# === Sacred Entrypoint ======================================
# ============================================================

@ex.main
def main(ghz_size, sweep, n_shots, num_runs, epsilon, notes, out_dir, _run):

    os.makedirs(out_dir, exist_ok=True)

    qc = make_ghz(ghz_size)
    results = []

    print("\n================ SWEEPING NOISE =================\n")

    for p1 in sweep["p_1q"]:
        for p2 in sweep["p_2q"]:
            for pm in sweep["p_meas"]:
                for ph in sweep["coherent_phase"]:

                    params = dict(
                        p_1q=p1,
                        p_2q=p2,
                        p_meas=pm,
                        coherent_phase=ph,
                        n_qubits=ghz_size,
                    )

                    print(f"Running sweep point: {params}")
                    noise_model = make_custom_noise_model(**params)

                    ENT_fid = run_single_tomo(
                        qc,
                        n_shots,
                        num_runs,
                        epsilon,
                        noise_model,
                        partial_mixing=False,
                    )

                    PM_fid = run_single_tomo(
                        qc,
                        n_shots,
                        num_runs,
                        epsilon,
                        noise_model,
                        partial_mixing=True,
                    )

                    print(f" → ENT fidelity = {ENT_fid:.6f}")
                    print(f" → PM  fidelity = {PM_fid:.6f}")

                    results.append({
                        "p_1q": p1,
                        "p_2q": p2,
                        "p_meas": pm,
                        "coherent_phase": ph,
                        "ENT_fidelity": ENT_fid,
                        "PM_fidelity": PM_fid,
                    })

    _run.info["results"] = results

    # --- produce full comparison plots ---
    plot_dual_heatmaps(results, sweep, out_dir, ghz_size)

    # Save CSV
    import csv
    out_csv = os.path.join(out_dir, "noise_sweep_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print("\nSweep results saved to:", out_csv)


if __name__ == "__main__":
    ex.run()
