# Efficient Circuit-Based Quantum State Tomography via Sparse Entry Optimization

Code accompanying the paper [Efficient Circuit-Based Quantum State Tomography via Sparse Entry Optimization](https://arxiv.org/abs/2407.20298). The repository holds all scripts used to synthesize circuits, run the Sacred-powered experiment sweeps, and regenerate the artifacts behind the plots in the manuscript.

## Environment
- Python: 3.10 (see `.python-version`)
- Dependency manager: `pip`

A lightweight setup looks like:

```bash
python -m venv .venv
source .venv/bin/activate
pip install qiskit qiskit-aer qiskit-ibm-runtime numpy scipy networkx sacred pyyaml matplotlib
```

## Repository layout
- `run_general_experiments.py`: Sacred entrypoint that replays batches of tomography jobs.
- `generate_circuit_files.py`: Rebuilds every circuit, metadata file, and Sacred config used to produce the paper’s figures; run this before reproducing plots.
- `experiments/configs/*`: One directory per experiment ID (QASM, metadata, config, standalone Sacred file).
- `experiments/batches/*.yaml`: Lists of experiment IDs that define each batch used in the paper.
- `experiments/runs*/`: Sacred run directories (logs, configs, reconstructions).
- `src/`: Tomography implementation (`algorithm.py`) and helpers (`measurements.py`, `qutils.py`, `noise_modeling.py`, `putils.py`).
- `plots/`, `paper_plotting.ipynb`: Notebooks/scripts to render the figures once the run data has been regenerated.

## Reproducing the simulations
1. **Generate experiment assets**
   ```bash
   python generate_circuit_files.py
   ```
   This refreshes `experiments/configs/<id>_auto/` with `circuit.qasm`, `metadata.yaml`, `config.yaml`, and `standalone_run.yaml`. These are the exact inputs the Sacred runner consumes.

2. **Select a batch definition**  
   Pick one of the YAML files under `experiments/batches/`. Each file lists the experiment IDs that correspond to a figure or table in the paper (e.g., `3q.yaml`, `ghz_chain.yaml`, `full.yaml`). You can inspect or create new batches as needed.

3. **Run the Sacred sweep**
   ```bash
   python run_general_experiments.py --batch-file experiments/batches/3q.yaml
   ```
   If `--batch-file` is omitted, the script interactively lists every batch. Sacred writes outputs to `experiments/runs/<run_id>/` where you will find `tomography.log`, the resolved Sacred configuration, and per-experiment metadata. Each experiment loads the matching circuit from `experiments/configs/<id>_auto/`, executes the tomography reconstruction with the requested number of shots and runs, and reports fidelities/errors in the log.

4. **Override runtime settings (optional)**  
   Any Sacred config knob can be modified via dotted updates. Examples:
   ```bash
   python run_general_experiments.py \
     --batch-file experiments/batches/ghz_chain.yaml \
     --update execution.n_shots=32768 execution.num_runs=64 \
              execution.noise_model.mode=custom \
              execution.noise_model.custom_params.p_1q=4.239e-4
   ```
   Common options include switching between `simulator`, `statevector`, or `ibm_qpu`, toggling `partial_mixing`, and adjusting the `noise_model` block (e.g., fake backend vs. custom error rates).

5. **Inspect results**  
   Each Sacred run directory contains:
   - `tomography.log`: fidelity/error summaries per experiment along with timing breakdowns.
   - `config.json`: the resolved Sacred configuration for traceability.
   - `info.json`: experiment metadata (including the batch IDs).

6. **Plotting**  
   Once the runs are complete, point `paper_plotting.ipynb` (or your own plotting scripts under `plots/`) at the matching Sacred run directories. Because `generate_circuit_files.py` rebuilds the precise QASM/config files used for the paper, the output statistics align with the original figures as long as the same batch definitions and seeds/noise models are used.

## Optional: IBM Quantum access
All paper results were produced on simulators, but the code can target IBM hardware. Before running on QPUs, save your IBM Quantum Platform token:

```ini
# config.ini
[IBM]
token = <YOUR_IBM_TOKEN>
```

```python
import configparser
from qiskit_ibm_runtime import QiskitRuntimeService

cp = configparser.ConfigParser()
cp.read("config.ini")
token = cp.get("IBM", "token")
QiskitRuntimeService.save_account(channel="ibm_quantum", token=token, overwrite=True)
```

Then set `execution.type=ibm_qpu` (and optionally target a specific backend via `execution.backend_name`) using the `--update` flag.

## Citation
If you use this repository, please cite the paper:

```
Chi-Kwong Li, Kevin Yipu Wu, and Zherui Zhang. “Minimum-Spanning-Tree Tomography of Sparse Quantum States
With and Without Entanglement.” arXiv:2407.20298, 29 July 2024 (revised 23 Dec 2025).
```
