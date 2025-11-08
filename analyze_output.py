#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml
from itertools import cycle

RUNS_DIR = "experiments/runs"
CONFIG_DIR = "plots/configs"

# --- Matplotlib style for papers ---
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (5, 3.5)
})


def get_experiment_label(exp_id, exp_labels, exp_pm, latex=False):
    base = exp_labels.get(exp_id, f"Exp {exp_id}")
    if latex and base != f"Exp {exp_id}":
        base = ", ".join([f"$x_{{{b}}}$" for b in base.split(", ")])
    if exp_pm.get(exp_id, False):
        base += r" (PM)" if not latex else r" (PM)"
    return base


# A palette that cycles nicely
COLOR_CYCLE = cycle(plt.cm.tab10.colors)

group_colors = {}


def get_group_color(exp_id):
    group_idx = exp_id // 2
    if group_idx not in group_colors:
        group_colors[group_idx] = next(COLOR_CYCLE)
    return group_colors[group_idx]


def get_marker(exp_id):
    return "^" if exp_id % 2 == 1 else "o"


def list_and_select(dir_glob, prompt="Select:"):
    files = sorted(glob.glob(dir_glob))
    if not files:
        raise FileNotFoundError(f"No files found matching {dir_glob}")
    print(f"Available options in {dir_glob}:")
    for i, f in enumerate(files, start=1):   # 1-indexed
        print(f"[{i}] {os.path.basename(f)}")
    while True:
        try:
            choice = int(input(f"{prompt} (1-{len(files)}): "))
            if 1 <= choice <= len(files):
                return files[choice-1]
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a number.")


def _tokens_to_complex_vector(tokens):
    """
    Convert a token list (possibly like ['0.', '+0.j', '0.70710678+0.j', ...])
    into a numpy complex vector. Handles both two-token and single-token forms.
    """
    vec = []
    prev = None
    for tok in tokens:
        if tok == '':
            continue
        # compact complex like '0.70710678+0.j' or '-1.2e-3-4e-2j'
        # ignore leading sign
        if 'j' in tok and ('+' in tok[1:] or '-' in tok[1:]):
            try:
                vec.append(complex(tok))
                prev = None
                continue
            except Exception:
                pass

        # two-token form: prev is real, current is like '+0.j' or '-1.2e-3j'
        if prev is not None and tok.endswith('j') and ('+' in tok or '-' in tok):
            try:
                c = complex(prev + tok)   # e.g. '0.' + '+0.j' -> '0.+0.j'
                vec.append(c)
                prev = None
                continue
            except Exception:
                # fall through and reset prev
                prev = None

        # if tok endswith 'j' but we didn't combine, try direct complex (e.g. '0.j')
        if tok.endswith('j'):
            try:
                vec.append(complex(tok))
                prev = None
                continue
            except Exception:
                prev = None
                continue

        # otherwise store as potential real-part token
        prev = tok

    return np.array(vec, dtype=complex)


def parse_experiment_fidelities(log_path):
    """
    Parse tomography.log for:
      - exp_data[exp_id]: list of fidelities
      - exp_labels[exp_id]: 'binary, binary, ...' for nonzero basis indices
      - exp_pm[exp_id]: bool (Partial mixing)
      - order: list of exp_ids in chronological appearance
    """
    exp_data = {}
    exp_labels = {}
    exp_pm = {}
    order = []

    current_exp = None
    capture_original = False
    original_buf = []

    with open(log_path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # Experiment ID
            m_exp = re.search(r"Experiment ID:\s+(\d+)", line)
            if m_exp:
                current_exp = int(m_exp.group(1))
                if current_exp not in exp_data:
                    exp_data[current_exp] = []
                    order.append(current_exp)
                capture_original = False
                original_buf = []
                continue

            # Partial mixing
            m_pm = re.search(r"Partial mixing:\s+(True|False)", line)
            if m_pm and current_exp is not None:
                exp_pm[current_exp] = (m_pm.group(1) == "True")
                continue

            # Start capturing original vector
            if "Original:" in line:
                capture_original = True
                original_buf = []
                continue

            # While capturing, accumulate until we see a closing ']'
            if capture_original:
                original_buf.append(line.strip())
                if ']' in line:
                    # Join, strip brackets/commas, split into tokens
                    joined = " ".join(original_buf)
                    inside = joined[joined.find('[')+1: joined.rfind(']')]
                    # Replace commas with spaces, normalize whitespace
                    token_str = inside.replace(",", " ")
                    tokens = token_str.split()
                    vec = _tokens_to_complex_vector(tokens)

                    # Sanity: if length isn't a power of two, warn
                    L = len(vec)
                    if L == 0 or (L & (L - 1)) != 0:
                        print(f"[warn] Parsed original state length {L} for exp {current_exp} "
                              f"is not a power of two. Check log formatting.")
                    else:
                        n_qubits = int(np.log2(L))
                        nz = np.where(np.abs(vec) > 1e-8)[0]
                        label = ", ".join(
                            format(i, f"0{n_qubits}b") for i in nz)
                        exp_labels[current_exp] = label

                    capture_original = False
                continue

            # Single fidelity lines
            m = re.search(r"Fidelity:\s+([0-9.]+)", line)
            if m and current_exp is not None:
                exp_data[current_exp].append(float(m.group(1)))
                continue

            # Bulk fidelities line
            m2 = re.search(r"Fidelities:\s+(.*)", line)
            if m2 and current_exp is not None:
                vals = [float(x) for x in m2.group(1).split(",")]
                exp_data[current_exp].extend(vals)
                continue

    return exp_data, exp_labels, exp_pm, order


def save_hist(data, exp_id, out_dir, bins=30):
    """Save histogram of infidelities with log-scaled x-axis (infidelity)."""
    if len(data) == 0:
        print(f"No fidelities for experiment {exp_id}")
        return

    infid = 1.0 - np.array(data)

    fig, ax = plt.subplots()
    ax.hist(infid, bins=bins, alpha=0.85, color="black", edgecolor="white")
    ax.set_xscale("log")   # <--- log scale for infidelity
    ax.set_xlabel("Infidelity (1 - F)")
    ax.set_ylabel("Count")
    ax.set_title(f"Experiment {exp_id}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_file = os.path.join(out_dir, f"exp_{exp_id}_infidelity_hist")
    fig.savefig(out_file + ".pdf")
    fig.savefig(out_file + ".png")
    plt.close(fig)

    print(f"Saved histogram for experiment {exp_id} → {out_file}.pdf/.png")


def plot_ecdf(ax, data, exp_id, label):
    infid = 1.0 - np.array(data)
    if len(infid) == 0:
        return
    infid_sorted = np.sort(infid)
    y = np.arange(1, len(infid_sorted) + 1) / len(infid_sorted)

    color = get_group_color(exp_id)
    marker = get_marker(exp_id)

    # Step line (not in legend)
    ax.step(infid_sorted, y, where="post", color=color)

    # Sparse markers (not in legend either)
    step = max(1, len(infid_sorted) // 20)
    ax.plot(
        infid_sorted[::step], y[::step],
        linestyle="none", marker=marker, color=color, markersize=5
    )

    # Add combined line+marker handle for legend
    handle = Line2D([0], [0],
                    color=color,
                    marker=marker,
                    linestyle="-",
                    markersize=6,
                    label=label)
    return handle


def compare_experiments(exp_data, exp_labels, exp_pm, order,
                        groups, out_dir, mode="violin"):
    """Overlay experiment groups as violin plots or ECDFs."""
    import matplotlib.pyplot as plt
    import numpy as np

    for g_idx, indices in enumerate(groups, start=1):
        fig, ax = plt.subplots()

        data = []
        labels = []
        for exp_id in indices:
            if exp_id not in exp_data:
                print(f"Experiment ID {exp_id} not found, skipping.")
                continue
            fids = exp_data[exp_id]
            data.append(fids)
            labels.append(get_experiment_label(
                exp_id, exp_labels, exp_pm, latex=True))

        if mode == "violin":
            # violin plot
            infid_data = [1.0 - np.array(f) for f in data]
            parts = ax.violinplot(
                infid_data,
                showmeans=True,
                showmedians=True,
                widths=0.8
            )
            # style violins
            for pc in parts['bodies']:
                pc.set_facecolor("lightgray")
                pc.set_edgecolor("black")
                pc.set_alpha(0.7)
            if "cbars" in parts:
                parts['cbars'].set_color("black")
            if "cmedians" in parts:
                parts['cmedians'].set_color("red")
            if "cmeans" in parts:
                parts['cmeans'].set_color("blue")

            ax.set_xticks(range(0, len(labels)))
            ax.set_xticklabels([get_experiment_label(order[idx], latex=True)
                               for idx in indices], rotation=30, ha="right")
            ax.set_yscale("log")
            ax.set_ylabel("Infidelity (1 - F)")
            ax.set_title(f"Violin Plot Comparison Group {g_idx}")

        elif mode == "ecdf":
            handles = []
            for fids, exp_id in zip(data, indices):
                label = get_experiment_label(
                    exp_id, exp_labels, exp_pm, latex=True)
                handle = plot_ecdf(ax, fids, exp_id, label)

                if handle:
                    handles.append(handle)

            ax.set_xscale("log")
            ax.set_xlabel("Infidelity (1 - F)")
            ax.set_ylabel("ECDF")
            ax.set_ylim(0, 1.0)
            ax.set_title(f"ECDF Comparison Group {g_idx}")

            # Legend with combined line+marker entries
            ax.legend(handles=handles, loc="upper left", title="Experiments")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        out_file = os.path.join(out_dir, f"{mode}_group{g_idx}_infidelity")
        fig.savefig(out_file + ".pdf")
        fig.savefig(out_file + ".png")
        plt.close(fig)
        print(f"Saved {mode} plot → {out_file}.pdf/.png")


def save_ecdf(data, exp_id, out_dir):
    """Save ECDF of infidelities with log-scaled x-axis (infidelity)."""
    if len(data) == 0:
        print(f"No fidelities for experiment {exp_id}")
        return

    infid = 1.0 - np.array(data)
    infid_sorted = np.sort(infid)
    y = np.arange(1, len(infid_sorted) + 1) / len(infid_sorted)

    fig, ax = plt.subplots()
    ax.step(infid_sorted, y, where="post", color="black")

    ax.set_xscale("log")
    ax.set_xlabel("Infidelity (1 - F)")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Experiment {exp_id}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_file = os.path.join(out_dir, f"exp_{exp_id}_infidelity_ecdf")
    fig.savefig(out_file + ".pdf")
    fig.savefig(out_file + ".png")
    plt.close(fig)

    print(f"Saved ECDF for experiment {exp_id} → {out_file}.pdf/.png")


def main():
    # --- Select run (1-indexed) ---
    run = list_and_select(os.path.join(RUNS_DIR, "[0-9]*"), "Select run")
    log_path = os.path.join(run, "tomography.log")
    if not os.path.exists(log_path):
        print(f"{log_path} not found.")
        return

    exp_data, exp_labels, exp_pm, order = parse_experiment_fidelities(log_path)
    print(f"Found {len(order)} experiments in this run:")
    for i, exp_id in enumerate(order, start=1):
        print(f"  [{i}] Experiment {exp_id} ({len(exp_data[exp_id])} fidelities)")

    # --- Select plotting config (1-indexed) ---
    cfg_file = list_and_select(os.path.join(
        CONFIG_DIR, "*.yaml"), "Select plotting config")
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f) or {}

    bins = cfg.get("bins", 30)
    compare_list = cfg.get("compare", None)
    mode = cfg.get("mode", "violin")  # default violin, can be "ecdf"

    # --- Output directory ---
    fig_dir = os.path.join(run, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if compare_list:
        compare_experiments(exp_data, exp_labels, exp_pm, order,
                            compare_list, fig_dir, mode=mode)
    else:
        for exp_id in order:
            save_hist(exp_data[exp_id], exp_id, fig_dir, bins=bins)
            save_ecdf(exp_data[exp_id], exp_id, fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze tomography run outputs with config selection.")
    args = parser.parse_args()
    main()
