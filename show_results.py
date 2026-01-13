#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Utils
# ============================================================

def get_final_value(npy_path):
    data = np.load(npy_path)
    if data.ndim > 1:
        return float(data[-1][0])
    return float(data[-1])

# ============================================================
# Core: mean / variance over runs
# ============================================================

def compute_cta_pta_mean_var(
    dataset,
    aggregator,
    budgets,
    runs,
    base_path=".",
    cta_file="caccs.npy",
    pta_file="paccs.npy",
    num_poisoned=1,
    num_clean=2,
    attack="backdoor",
):
    records = []

    for budget in budgets:
        cta_vals, pta_vals = [], []

        for run in runs:
            run_dir = os.path.join(
                base_path,
                f"out/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}",
                str(run),
                str(budget),
            )

            cta_path = os.path.join(run_dir, cta_file)
            pta_path = os.path.join(run_dir, pta_file)

            if not (os.path.exists(cta_path) and os.path.exists(pta_path)):
                print(f"[WARNING] Missing files in {run_dir}")
                continue

            cta_vals.append(get_final_value(cta_path))
            pta_vals.append(get_final_value(pta_path))

        records.append({
            "dataset": dataset,
            "aggregator": aggregator,
            "budget": budget,
            "cta_mean": np.mean(cta_vals) if cta_vals else np.nan,
            "cta_var": np.var(cta_vals) if cta_vals else np.nan,
            "pta_mean": np.mean(pta_vals) if pta_vals else np.nan,
            "pta_var": np.var(pta_vals) if pta_vals else np.nan,
            "n_runs": len(cta_vals),
        })

    return pd.DataFrame.from_records(records)

# ============================================================
# Plot mean / var
# ============================================================

def plot_cta_vs_pta_mean_var(df, save_dir=None):
    plt.figure(figsize=(8, 6))

    plt.errorbar(
        df["pta_mean"],
        df["cta_mean"],
        xerr=np.sqrt(df["pta_var"]),
        yerr=np.sqrt(df["cta_var"]),
        fmt="o-",
        capsize=4,
        linewidth=2,
    )

    for _, row in df.iterrows():
        plt.text(
            row["pta_mean"] + 0.002,
            row["cta_mean"] + 0.002,
            str(int(row["budget"])),
            fontsize=9,
        )

    dataset_title = "SVHN" if df.iloc[0]["dataset"].lower() == "svhn" else "CIFAR-10"
    plt.title(f"{dataset_title} – {df.iloc[0]['aggregator'].upper()} (mean ± std)")
    plt.xlabel("Poisoned Test Accuracy (PTA)")
    plt.ylabel("Clean Test Accuracy (CTA)")
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "cta_vs_pta_mean_var.png")
        plt.savefig(path)
        print(f"[INFO] Saved plot: {path}")


# ============================================================
# Per-run utilities
# ============================================================

def collect_cta_pta_per_run(
    dataset,
    aggregator,
    budgets,
    run,
    base_path=".",
    cta_file="caccs.npy",
    pta_file="paccs.npy",
    num_poisoned=1,
    num_clean=2,
    attack="backdoor",
):
    records = []

    for budget in budgets:
        run_dir = os.path.join(
            base_path,
            f"out/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}",
            str(run),
            str(budget),
        )

        cta_path = os.path.join(run_dir, cta_file)
        pta_path = os.path.join(run_dir, pta_file)

        if not (os.path.exists(cta_path) and os.path.exists(pta_path)):
            continue

        records.append({
            "run": run,
            "budget": budget,
            "cta": get_final_value(cta_path),
            "pta": get_final_value(pta_path),
        })

    return pd.DataFrame.from_records(records)

def plot_cta_vs_pta_single_run(df, dataset, aggregator, run, save_dir=None, num_poisoned=1, num_clean=2, attack="backdoor"):
    plt.figure(figsize=(7, 6))

    plt.plot(df["pta"], df["cta"], "o-", linewidth=2)

    for _, row in df.iterrows():
        plt.text(row["pta"] + 0.002, row["cta"] + 0.002, str(int(row["budget"])))

    plt.title(f"{dataset.upper()} – {aggregator.upper()} – Run {run}")
    plt.xlabel("Poisoned Test Accuracy (PTA)")
    plt.ylabel("Clean Test Accuracy (CTA)")
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"cta_vs_pta_run_{run}.png")
        plt.savefig(path)
        print(f"[INFO] Saved plot: {path}")

def plot_cta_vs_pta_multi_aggregators(
    dfs_by_agg,
    dataset,
    save_path=None,
):
    """
    dfs_by_agg: dict {aggregator_name: DataFrame}
    """

    plt.figure(figsize=(8.5, 7))

    # Styles fixes (faciles à comparer)
    markers = ["o", "s", "^", "D", "v"]
    linestyles = ["-", "--", "-.", ":", "-"]

    for i, (agg, df) in enumerate(dfs_by_agg.items()):
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]

        budgets = df["budget"].values
        x = df["pta_mean"].values * 100
        y = df["cta_mean"].values * 100

        # Annotate budgets
        for b, xi, yi in zip(budgets, x, y):
            plt.text(
                xi + 0.8,
                yi + 0.3,
                str(int(b)),
                fontsize=9,
                alpha=0.8,
            )

        plt.errorbar(
            x,
            y,
            xerr=np.sqrt(df["pta_var"].values) * 100,
            yerr=np.sqrt(df["cta_var"].values) * 100,
            linestyle=linestyle,
            linewidth=2.0,
            marker=marker,
            markersize=7,
            markeredgecolor="black",
            capsize=4,
            elinewidth=1.6,
            alpha=0.9,
            label=f"{agg.capitalize()} aggregation",
        )

    # -----------------------------
    # Axes & style
    # -----------------------------
    plt.xlabel("Poisoned Test Accuracy (%)", fontsize=13)
    plt.ylabel("Clean Test Accuracy (%)", fontsize=13)

    # plt.xlim(0, 100)
    # plt.ylim(65, 94)

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(frameon=True, fontsize=11)

    dataset_title = "SVHN" if dataset.lower() == "svhn" else "CIFAR-10"
    plt.title(f"{dataset_title}: Clean vs Poisoned Test Accuracy", fontsize=15)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Saved plot: {save_path}")

    plt.show()


def plot_cta_vs_pta_single_run_multi_aggregators(
    dfs_by_agg_run,
    dataset,
    run,
    save_path=None,
):
    """
    dfs_by_agg_run: dict {aggregator: DataFrame for ONE run}
    """

    plt.figure(figsize=(8, 7))

    markers = ["o", "s", "^", "D", "v"]
    linestyles = ["-", "--", "-.", ":", "-"]

    for i, (agg, df) in enumerate(dfs_by_agg_run.items()):
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]

        x = df["pta"].values * 100
        y = df["cta"].values * 100
        budgets = df["budget"].values

        for b, xi, yi in zip(budgets, x, y):
            plt.text(xi + 0.6, yi + 0.3, str(int(b)), fontsize=8, alpha=0.7)

        plt.plot(
            x,
            y,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            markersize=7,
            markeredgecolor="black",
            label=f"{agg.capitalize()} aggregation",
        )

    plt.xlabel("Poisoned Test Accuracy (%)", fontsize=13)
    plt.ylabel("Clean Test Accuracy (%)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(frameon=True, fontsize=11)

    dataset_title = "SVHN" if dataset.lower() == "svhn" else "CIFAR-10"
    plt.title(f"{dataset_title} – Aggregator comparison (Run {run})", fontsize=15)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Saved plot: {save_path}")

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    DATASET = "cifar"  # or "svhn"
    AGGREGATORS = ["mean", "median"]
    BUDGETS = [150, 300, 500, 1000, 2000, 2500, 5000, 10000]
    RUNS = range(5, 15)

    BASE_PATH = "."
    OUTPUT_DIR = "./plots"
    CSV_DIR = "./results_csv"
    RUN_PLOTS_DIR = "./plots_per_run"
    NUM_POISONED = 1
    NUM_CLEAN = 2
    ATTACK = "backdoor"

    os.makedirs(CSV_DIR, exist_ok=True)

    # ==============================
    # Collect mean/var per aggregator
    # ==============================
    dfs_by_agg = {}

    for aggregator in AGGREGATORS:
        print(f"\n=== Processing {DATASET} / {aggregator} ===")

        df_mean = compute_cta_pta_mean_var(
            dataset=DATASET,
            aggregator=aggregator,
            budgets=BUDGETS,
            runs=RUNS,
            base_path=BASE_PATH,
            num_poisoned=NUM_POISONED,
            num_clean=NUM_CLEAN,
            attack=ATTACK,
        )

        dfs_by_agg[aggregator] = df_mean

        # Save CSV
        csv_path = os.path.join(
            CSV_DIR, f"{DATASET}_{aggregator}_cta_pta_mean_var.csv"
        )
        df_mean.to_csv(csv_path, index=False)
        print(f"[INFO] Saved CSV: {csv_path}")

        for run in RUNS:
            dfs_run_by_agg = {}

            for aggregator in AGGREGATORS:
                df_run = collect_cta_pta_per_run(
                    dataset=DATASET,
                    aggregator=aggregator,
                    budgets=BUDGETS,
                    run=run,
                    base_path=BASE_PATH,
                    num_poisoned=NUM_POISONED,
                    num_clean=NUM_CLEAN,
                    attack=ATTACK,
                )

                if not df_run.empty:
                    dfs_run_by_agg[aggregator] = df_run

            if len(dfs_run_by_agg) >= 2:
                plot_cta_vs_pta_single_run_multi_aggregators(
                    dfs_run_by_agg,
                    dataset=DATASET,
                    run=run,
                    save_path=f"./plots/{DATASET}_cta_vs_pta_run_{run}_agg_comparison.png",
                )

    # ==============================
    # Single comparison plot
    # ==============================
    plot_cta_vs_pta_multi_aggregators(
        dfs_by_agg,
        dataset=DATASET,
        save_path=f"./plots/{DATASET}_cta_vs_pta_aggregator_comparison.png",
    )
