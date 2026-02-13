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
# Mean / variance over runs — ONE ATTACK
# ============================================================

def compute_cta_pta_mean_var(
    dataset,
    attack,
    budgets,
    runs,
    base_path=".",
    cta_file="caccs.npy",
    pta_file="paccs.npy",
    num_poisoned=1,
    num_clean=2,
):
    records = []

    for budget in budgets:
        cta_vals, pta_vals = [], []

        for run in runs:

            run_dir = os.path.join(
                base_path,
                f"out/{num_poisoned}vs{num_clean}/{dataset}_{attack}",
                str(budget),
                str(run),
            )

            cta_path = os.path.join(run_dir, cta_file)
            pta_path = os.path.join(run_dir, pta_file)

            if not (os.path.exists(cta_path) and os.path.exists(pta_path)):
                print(f"[WARNING] Missing files in {run_dir}")
                continue

            cta_vals.append(get_final_value(cta_path))
            pta_vals.append(get_final_value(pta_path))

        records.append({
            "attack": attack,
            "budget": budget,
            "cta_mean": np.mean(cta_vals) if cta_vals else np.nan,
            "cta_var": np.var(cta_vals) if cta_vals else np.nan,
            "pta_mean": np.mean(pta_vals) if pta_vals else np.nan,
            "pta_var": np.var(pta_vals) if pta_vals else np.nan,
            "n_runs": len(cta_vals),
        })

    return pd.DataFrame.from_records(records)


# ============================================================
# Plot comparison between attacks
# ============================================================

def plot_cta_vs_pta_multi_attacks(
    dfs_by_attack,
    dataset,
    save_path=None,
):

    plt.figure(figsize=(8.5, 7))

    markers = ["o", "s", "^", "D", "v", "P"]
    linestyles = ["-", "--", "-.", ":", "-"]

    for i, (attack, df) in enumerate(dfs_by_attack.items()):

        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]

        x = df["pta_mean"].values * 100
        y = df["cta_mean"].values * 100
        budgets = df["budget"].values

        # annotate budgets
        for b, xi, yi in zip(budgets, x, y):
            plt.text(xi + 0.5, yi + 0.3, str(int(b)), fontsize=9)

        plt.errorbar(
            x,
            y,
            xerr=np.sqrt(df["pta_var"].values) * 100,
            yerr=np.sqrt(df["cta_var"].values) * 100,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            markersize=7,
            markeredgecolor="black",
            capsize=4,
            label=attack,
        )

    dataset_title = "SVHN" if dataset.lower() == "svhn" else "CIFAR-10"

    plt.xlabel("Poisoned Test Accuracy (%)", fontsize=13)
    plt.ylabel("Clean Test Accuracy (%)", fontsize=13)
    plt.title(f"{dataset_title} — Attack comparison", fontsize=15)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Saved plot: {save_path}")

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    DATASET = "cifar"
    ATTACKS = ["1xs", "1xp", "4xl", "opti"]

    BUDGETS = [150, 300, 500, 1000, 2000, 2500, 5000]
    RUNS = range(1, 6)

    BASE_PATH = "."
    NUM_POISONED = 1
    NUM_CLEAN = 0

    CSV_DIR = "./results_csv"
    PLOT_DIR = "./plots"

    os.makedirs(CSV_DIR, exist_ok=True)

    dfs_by_attack = {}

    print("\n=== Processing attacks ===")

    # -------------------------------------------------
    # LOOP OVER ATTACKS
    # -------------------------------------------------

    for attack in ATTACKS:

        print(f"\n--- Attack: {attack} ---")

        df = compute_cta_pta_mean_var(
            dataset=DATASET,
            attack=attack,
            budgets=BUDGETS,
            runs=RUNS,
            base_path=BASE_PATH,
            num_poisoned=NUM_POISONED,
            num_clean=NUM_CLEAN,
        )

        dfs_by_attack[attack] = df

        csv_path = os.path.join(
            CSV_DIR,
            f"{DATASET}_{attack}_cta_pta_mean_var.csv",
        )
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved CSV: {csv_path}")

    # -------------------------------------------------
    # GLOBAL COMPARISON PLOT
    # -------------------------------------------------

    plot_cta_vs_pta_multi_attacks(
        dfs_by_attack,
        dataset=DATASET,
        save_path=f"{PLOT_DIR}/{DATASET}_attack_comparison.png",
    )
