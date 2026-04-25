#!/usr/bin/env python3
"""
Exercise 3 — Dynamic analysis of CUDA vector-sum kernels.

Compiles both binaries, runs all parameter combinations, saves results to
CSV, and generates plots for the performance report.

Usage:
    python3 run_exercise3.py                  # full run
    python3 run_exercise3.py --skip-bench     # reuse existing CSV, only plot
    python3 run_exercise3.py --runs 3         # fewer repetitions for a quick test
"""

import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import argparse
import csv
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
PLOTS = RESULTS / "plots"
CSV_PATH = RESULTS / "exercise3.csv"

GPU_BIN = BASE / "sumVectors"
SEQ_BIN = BASE / "sumVectorsSec"
GPU_SRC = BASE / "sumVectors.cu"
SEQ_SRC = BASE / "sumVectors.c"
COMMON = BASE.parent / "cuda-samples" / "Common"

# ── Experiment parameters ─────────────────────────────────────────────────────
N_VALUES = [2**i for i in range(16, 26)]   # 2^16 … 2^25  (10 values)
COMP_VALUES = [1, 64, 256]                    # elements per thread  (m)
BLOCK_SIZES = [16, 32, 64, 128, 256]          # threads per block
KERNELS = ["a", "b"]
N_RUNS = 10                              # repetitions per configuration

# ── Plot style ────────────────────────────────────────────────────────────────
KERNEL_COLOR = {"a": "#1f77b4", "b": "#ff7f0e", "seq": "#2ca02c"}
KERNEL_LABEL = {"a": "Kernel A", "b": "Kernel B", "seq": "Sequential"}
plt.rcParams.update({"figure.dpi": 140, "font.size": 9})


# ─────────────────────────────────────────────────────────────────────────────
# Compilation
# ─────────────────────────────────────────────────────────────────────────────
def compile_all():
    print("Compiling GPU binary …", flush=True)
    r = subprocess.run(
        ["nvcc", str(GPU_SRC), "-o", str(GPU_BIN),
         "--ptxas-options=-v",
         f"-I{COMMON}", f"-L{COMMON}/lib"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"GPU compile failed:\n{r.stderr}")

    print("Compiling sequential binary …", flush=True)
    r = subprocess.run(
        ["gcc", str(SEQ_SRC), "-o", str(SEQ_BIN), "-O2"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"Sequential compile failed:\n{r.stderr}")

    print("Compilation done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Running
# ─────────────────────────────────────────────────────────────────────────────
def run_gpu(n, comp, kernel, block_size):
    """Returns (kernel_time_s, total_time_s) or (None, None) on failure."""
    r = subprocess.run(
        [str(GPU_BIN), str(n), str(comp), kernel, "0", str(block_size)],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        return None, None
    kt = re.search(r"Kernel time:\s*([\d.eE+\-]+)", r.stdout)
    tt = re.search(r"Total time:\s*([\d.eE+\-]+)", r.stdout)
    return (float(kt.group(1)) if kt else None,
            float(tt.group(1)) if tt else None)


def run_seq(n):
    """Returns total_time_s or None on failure."""
    r = subprocess.run(
        [str(SEQ_BIN), str(n)],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        return None
    m = re.search(r"Time\s*=\s*([\d.eE+\-]+)", r.stdout)
    return float(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking
# ─────────────────────────────────────────────────────────────────────────────
def benchmark(n_runs):
    rows = []

    gpu_configs = [
        (kernel, comp, bs, n)
        for kernel in KERNELS
        for comp in COMP_VALUES
        for bs in BLOCK_SIZES
        for n in N_VALUES
    ]
    total = len(gpu_configs) * n_runs + len(N_VALUES) * n_runs
    done = 0

    print(
        f"Total configurations: {len(gpu_configs)} GPU + {len(N_VALUES)} sequential")
    print(
        f"Total runs: {total}\n")

    for kernel, comp, bs, n in gpu_configs:
        for run in range(1, n_runs + 1):
            kt, tt = run_gpu(n, comp, kernel, bs)
            rows.append({
                "kernel":      kernel,
                "n":           n,
                "comp":        comp,
                "block_size":  bs,
                "run":         run,
                "kernel_time": kt,
                "total_time":  tt,
            })
            done += 1
            exp = int(math.log2(n))
            print(f"\r  [{done}/{total}]  kernel={kernel}  comp={comp:>3}  "
                  f"bs={bs:>3}  n=2^{exp}  run={run}   ", end="", flush=True)

    for n in N_VALUES:
        for run in range(1, n_runs + 1):
            t = run_seq(n)
            rows.append({
                "kernel":      "seq",
                "n":           n,
                "comp":        None,
                "block_size":  None,
                "run":         run,
                "kernel_time": None,
                "total_time":  t,
            })
            done += 1
            exp = int(math.log2(n))
            print(f"\r  [{done}/{total}]  seq  n=2^{exp}  run={run}              ",
                  end="", flush=True)

    print(f"\r  [{done}/{total}]  done.                                      ")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────
FIELDS = ["kernel", "n", "comp", "block_size",
          "run", "kernel_time", "total_time"]


def save_csv(rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row[k] is None else row[k]) for k in FIELDS})
    print(f"\nResults saved → {CSV_PATH}")


def load_csv():
    def _cast(row):
        row["n"] = int(row["n"])
        row["run"] = int(row["run"])
        row["comp"] = int(row["comp"]) if row["comp"] else None
        row["block_size"] = int(
            row["block_size"]) if row["block_size"] else None
        row["kernel_time"] = float(
            row["kernel_time"]) if row["kernel_time"] else None
        row["total_time"] = float(
            row["total_time"]) if row["total_time"] else None
        return row

    with open(CSV_PATH) as f:
        rows = [_cast(r) for r in csv.DictReader(f)]
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────
def avg_by(rows, group_keys, value_key):
    """Group rows by group_keys and return average of value_key per group."""
    buckets = defaultdict(list)
    for r in rows:
        v = r[value_key]
        if v is not None:
            buckets[tuple(r[k] for k in group_keys)].append(v)
    return {k: sum(vs) / len(vs) for k, vs in buckets.items()}


def n_label(n):
    return f"$2^{{{int(math.log2(n))}}}$"


# ─────────────────────────────────────────────────────────────────────────────
# Shared axis setup
# ─────────────────────────────────────────────────────────────────────────────
def _decorate(ax, title, ylabel="Time (s)"):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Vector size n")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: n_label(v) if v > 0 else "")
    )
    ax.legend(fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Kernel A vs B vs Sequential
#   For each (comp, block_size): left = kernel time, right = total time
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(rows):
    out = PLOTS / "01_comparison"
    out.mkdir(parents=True, exist_ok=True)

    avg_kt = avg_by(rows, ["kernel", "n", "comp", "block_size"], "kernel_time")
    avg_tt = avg_by(rows, ["kernel", "n", "comp", "block_size"], "total_time")
    seq_tt = avg_by([r for r in rows if r["kernel"]
                    == "seq"], ["n"], "total_time")

    for comp in COMP_VALUES:
        for bs in BLOCK_SIZES:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
            fig.suptitle(
                f"Kernel A vs B vs Sequential — m={comp}, threads/block={bs}",
                fontsize=10,
            )

            for ax, avg, include_seq, title in [
                (ax1, avg_kt, False, "Kernel time"),
                (ax2, avg_tt, True,  "Total time"),
            ]:
                for k in KERNELS:
                    pts = [(n, avg[(k, n, comp, bs)]) for n in N_VALUES
                           if (k, n, comp, bs) in avg]
                    if pts:
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys, "o-", color=KERNEL_COLOR[k],
                                label=KERNEL_LABEL[k])
                if include_seq:
                    pts = [(n, seq_tt[(n,)])
                           for n in N_VALUES if (n,) in seq_tt]
                    if pts:
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys, "s--", color=KERNEL_COLOR["seq"],
                                label=KERNEL_LABEL["seq"])
                _decorate(ax, title)

            fig.tight_layout()
            fig.savefig(out / f"m{comp}_bs{bs}.png")
            plt.close(fig)

    print(f"  [01] Comparison plots → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Effect of threads/block
#   For each (kernel, comp): one line per block_size, kernel time vs n
# ─────────────────────────────────────────────────────────────────────────────
def plot_blocksize_effect(rows):
    out = PLOTS / "02_blocksize_effect"
    out.mkdir(parents=True, exist_ok=True)

    avg_kt = avg_by(rows, ["kernel", "n", "comp", "block_size"], "kernel_time")
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(BLOCK_SIZES)))

    for kernel in KERNELS:
        for comp in COMP_VALUES:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for i, bs in enumerate(BLOCK_SIZES):
                pts = [(n, avg_kt[(kernel, n, comp, bs)]) for n in N_VALUES
                       if (kernel, n, comp, bs) in avg_kt]
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, "o-", color=colors[i], label=f"bs={bs}")
            _decorate(ax,
                      f"Effect of threads/block — Kernel {kernel.upper()}, m={comp}",
                      ylabel="Kernel time (s)")
            fig.tight_layout()
            fig.savefig(out / f"kernel{kernel}_m{comp}.png")
            plt.close(fig)

    print(f"  [02] Block-size effect plots → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Effect of m (comp)
#   For each (kernel, block_size): one line per comp value, kernel time vs n
# ─────────────────────────────────────────────────────────────────────────────
def plot_comp_effect(rows):
    out = PLOTS / "03_comp_effect"
    out.mkdir(parents=True, exist_ok=True)

    avg_kt = avg_by(rows, ["kernel", "n", "comp", "block_size"], "kernel_time")
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    for kernel in KERNELS:
        for bs in BLOCK_SIZES:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for i, comp in enumerate(COMP_VALUES):
                pts = [(n, avg_kt[(kernel, n, comp, bs)]) for n in N_VALUES
                       if (kernel, n, comp, bs) in avg_kt]
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, "o-", color=colors[i], label=f"m={comp}")
            _decorate(ax,
                      f"Effect of m — Kernel {kernel.upper()}, threads/block={bs}",
                      ylabel="Kernel time (s)")
            fig.tight_layout()
            fig.savefig(out / f"kernel{kernel}_bs{bs}.png")
            plt.close(fig)

    print(f"  [03] Comp effect plots → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Speedup over sequential (total time)
#   For each comp: subplot per kernel, one line per block_size
# ─────────────────────────────────────────────────────────────────────────────
def plot_speedup(rows):
    out = PLOTS / "04_speedup"
    out.mkdir(parents=True, exist_ok=True)

    avg_tt = avg_by(rows, ["kernel", "n", "comp", "block_size"], "total_time")
    seq_tt = avg_by([r for r in rows if r["kernel"]
                    == "seq"], ["n"], "total_time")
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(BLOCK_SIZES)))

    for comp in COMP_VALUES:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(f"GPU speedup over sequential — m={comp}", fontsize=10)

        for ax, kernel in zip(axes, KERNELS):
            for i, bs in enumerate(BLOCK_SIZES):
                pts = []
                for n in N_VALUES:
                    gpu = avg_tt.get((kernel, n, comp, bs))
                    seq = seq_tt.get((n,))
                    if gpu and seq and gpu > 0:
                        pts.append((n, seq / gpu))
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, "o-", color=colors[i], label=f"bs={bs}")

            ax.axhline(y=1, color="k", linestyle="--", linewidth=0.8,
                       alpha=0.6, label="Seq baseline")
            ax.set_title(f"Kernel {kernel.upper()}", fontsize=9)
            ax.set_xlabel("Vector size n")
            ax.set_ylabel("Speedup")
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: n_label(v) if v > 0 else "")
            )
            ax.legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(out / f"speedup_m{comp}.png")
        plt.close(fig)

    print(f"  [04] Speedup plots → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Best configuration summary
#   For each kernel: kernel time vs n for the config with lowest avg kernel time,
#   with all three comp values shown. Useful as a single report figure.
# ─────────────────────────────────────────────────────────────────────────────
def plot_best_config(rows):
    out = PLOTS / "05_best_config"
    out.mkdir(parents=True, exist_ok=True)

    avg_kt = avg_by(rows, ["kernel", "n", "comp", "block_size"], "kernel_time")
    seq_tt = avg_by([r for r in rows if r["kernel"]
                    == "seq"], ["n"], "total_time")
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Best configuration per kernel (lowest avg kernel time)", fontsize=10)

    for ax, kernel in zip(axes, KERNELS):
        # Find (comp, bs) with lowest average kernel time across all n
        best, best_key = float("inf"), None
        for comp in COMP_VALUES:
            for bs in BLOCK_SIZES:
                vals = [avg_kt.get((kernel, n, comp, bs)) for n in N_VALUES]
                vals = [v for v in vals if v is not None]
                if vals:
                    mean = sum(vals) / len(vals)
                    if mean < best:
                        best, best_key = mean, (comp, bs)

        if best_key:
            comp_b, bs_b = best_key
            ax.set_title(
                f"Kernel {kernel.upper()} — best: m={comp_b}, bs={bs_b}", fontsize=9
            )

        for i, comp in enumerate(COMP_VALUES):
            bs = best_key[1] if best_key else BLOCK_SIZES[0]
            pts = [(n, avg_kt[(kernel, n, comp, bs)]) for n in N_VALUES
                   if (kernel, n, comp, bs) in avg_kt]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=colors[i], label=f"m={comp}")

        # Sequential reference
        pts = [(n, seq_tt[(n,)]) for n in N_VALUES if (n,) in seq_tt]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "s--",
                    color=KERNEL_COLOR["seq"], label="Sequential")

        _decorate(ax, ax.get_title(), ylabel="Time (s)")

    fig.tight_layout()
    fig.savefig(out / "best_config.png")
    plt.close(fig)
    print(f"  [05] Best-config summary → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Summary (3 consolidated figures)
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary(rows):
    out = PLOTS / "06_summary"
    out.mkdir(parents=True, exist_ok=True)

    avg_kt = avg_by(rows, ["kernel", "n", "comp", "block_size"], "kernel_time")
    avg_tt = avg_by(rows, ["kernel", "n", "comp", "block_size"], "total_time")
    seq_tt = avg_by([r for r in rows if r["kernel"]
                    == "seq"], ["n"], "total_time")
    bs_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(BLOCK_SIZES)))

    # ── Figure 1: time scaling ────────────────────────────────────────────────
    # 2 rows (kernel A/B) × 3 cols (m=1,64,256). Lines = block_sizes + seq ref.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.suptitle("Kernel time vs n — all configurations", fontsize=11)

    for row_i, kernel in enumerate(KERNELS):
        for col_i, comp in enumerate(COMP_VALUES):
            ax = axes[row_i][col_i]
            for i, bs in enumerate(BLOCK_SIZES):
                pts = [(n, avg_kt[(kernel, n, comp, bs)]) for n in N_VALUES
                       if (kernel, n, comp, bs) in avg_kt]
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, "o-", color=bs_colors[i],
                            linewidth=1.2, label=f"bs={bs}")
            # Sequential reference (total time, dashed grey)
            pts = [(n, seq_tt[(n,)]) for n in N_VALUES if (n,) in seq_tt]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "--", color="0.55", linewidth=1,
                        label="Sequential")
            ax.set_title(f"Kernel {kernel.upper()}, m={comp}", fontsize=9)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.2, linestyle="--")
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: n_label(v) if v > 0 else ""))
            if col_i == 0:
                ax.set_ylabel("Kernel time (s)")
            if row_i == 1:
                ax.set_xlabel("Vector size n")
            ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(out / "time_scaling.png")
    plt.close(fig)

    # ── Figure 2: configuration heatmaps ─────────────────────────────────────
    # 2 rows (kernel A/B) × 2 cols (kernel time / total time).
    # Each cell: heatmap of comp (rows) × block_size (cols), avg over large n.
    large_n = N_VALUES[len(N_VALUES) // 2:]   # upper half of n values
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Average time heatmap (comp × threads/block)", fontsize=11)

    for row_i, kernel in enumerate(KERNELS):
        for col_i, (avg, metric) in enumerate([(avg_kt, "Kernel time (s)"),
                                               (avg_tt, "Total time (s)")]):
            ax = axes[row_i][col_i]
            data = np.zeros((len(COMP_VALUES), len(BLOCK_SIZES)))
            for ci, comp in enumerate(COMP_VALUES):
                for bi, bs in enumerate(BLOCK_SIZES):
                    vals = [avg.get((kernel, n, comp, bs)) for n in large_n]
                    vals = [v for v in vals if v is not None]
                    data[ci, bi] = sum(vals) / len(vals) if vals else np.nan

            im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
            ax.set_xticks(range(len(BLOCK_SIZES)))
            ax.set_xticklabels(BLOCK_SIZES)
            ax.set_yticks(range(len(COMP_VALUES)))
            ax.set_yticklabels(COMP_VALUES)
            ax.set_xlabel("Threads per block")
            ax.set_ylabel("m (comp)")
            ax.set_title(f"Kernel {kernel.upper()} — {metric}", fontsize=9)
            # Annotate cells with the value
            for ci in range(len(COMP_VALUES)):
                for bi in range(len(BLOCK_SIZES)):
                    v = data[ci, bi]
                    if not np.isnan(v):
                        ax.text(bi, ci, f"{v:.5f}", ha="center", va="center",
                                fontsize=7,
                                color="white" if v > np.nanmedian(data) else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out / "config_heatmap.png")
    plt.close(fig)

    # ── Figure 3: speedup A vs B side by side ────────────────────────────────
    # Panels: one per kernel. Lines = block_sizes. Uses best comp per kernel.
    # Find best comp for each kernel (lowest avg total time over large n)
    best_comp = {}
    for kernel in KERNELS:
        best, best_c = float("inf"), COMP_VALUES[0]
        for comp in COMP_VALUES:
            vals = [avg_tt.get((kernel, n, comp, bs))
                    for n in large_n for bs in BLOCK_SIZES]
            vals = [v for v in vals if v is not None]
            if vals and (m := sum(vals) / len(vals)) < best:
                best, best_c = m, comp
        best_comp[kernel] = best_c

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Speedup over sequential (total time)", fontsize=11)

    for ax, kernel in zip(axes, KERNELS):
        comp = best_comp[kernel]
        for i, bs in enumerate(BLOCK_SIZES):
            pts = []
            for n in N_VALUES:
                gpu = avg_tt.get((kernel, n, comp, bs))
                seq = seq_tt.get((n,))
                if gpu and seq and gpu > 0:
                    pts.append((n, seq / gpu))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=bs_colors[i], label=f"bs={bs}")
        ax.axhline(y=1, color="k", linestyle="--", linewidth=0.8,
                   alpha=0.5, label="Seq baseline")
        ax.set_title(f"Kernel {kernel.upper()} (m={comp})", fontsize=9)
        ax.set_xlabel("Vector size n")
        ax.set_ylabel("Speedup")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: n_label(v) if v > 0 else ""))
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out / "speedup.png")
    plt.close(fig)

    print(f"  [06] Summary figures → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-bench", action="store_true",
                        help="Skip benchmarking; load existing CSV and only plot")
    parser.add_argument("--skip-compile", action="store_true",
                        help="Skip compilation (use existing binaries)")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Repetitions per configuration (default: {N_RUNS})")
    args = parser.parse_args()

    if not args.skip_bench:
        if not args.skip_compile:
            compile_all()
        rows = benchmark(args.runs)
        save_csv(rows)
    else:
        if not CSV_PATH.exists():
            sys.exit(
                f"CSV not found: {CSV_PATH}\nRun without --skip-bench first.")
        rows = load_csv()

    print("\nGenerating plots …")
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_comparison(rows)
    plot_blocksize_effect(rows)
    plot_comp_effect(rows)
    plot_speedup(rows)
    plot_best_config(rows)
    plot_summary(rows)
    print(f"\nAll plots saved under {PLOTS}/")


if __name__ == "__main__":
    main()
