"""Run the complete pipeline:  raw CSV -> clean CSV -> model-ready train/test files.

Usage (from the repository root):
    python src/run_pipeline.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cleaning
import config
import evaluate
import generate_data
import preprocessing


def make_figures(snaps, clean_df, metrics):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    green, grey = "#2E7D32", "#9E9E9E"
    # Figure 1: missing values before / after
    m = metrics["missing_pct"]
    cols = list(m)
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, [m[c]["before"] for c in cols], 0.4, label="Before cleaning", color=grey)
    ax.bar(x + 0.2, [m[c]["after"] for c in cols], 0.4, label="After cleaning", color=green)
    ax.set_xticks(x); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_ylabel("Missing values (%)"); ax.set_title("Missing values before and after cleaning")
    ax.legend(); fig.tight_layout(); fig.savefig(config.FIG_DIR / "missing_before_after.png", dpi=150); plt.close(fig)

    # Figure 2: yield by crop, before / after outlier treatment
    crops = sorted(clean_df["Crop"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, df, title in [(axes[0], snaps["before_outliers"], "Before outlier treatment"),
                          (axes[1], clean_df, "After outlier treatment")]:
        ax.boxplot([df.loc[df.Crop == c, "Yield_q_ha"] for c in crops], tick_labels=crops,
                   flierprops=dict(marker="o", markerfacecolor="#C62828", markersize=4, markeredgecolor="none"))
        ax.set_title(title); ax.set_ylabel("Yield (quintal per ha)"); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "yield_outliers_before_after.png", dpi=150); plt.close(fig)

    # Figure 3: imputation error, group median vs simple baselines (rainfall)
    r = metrics["imputation"]["Rainfall_mm"]
    names = ["Global mean", "Global median", "Group median\n(District + Season)"]
    vals = [r["mae_global_mean"], r["mae_global_median"], r["mae_group_median_(used)"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, vals, color=[grey, grey, green])
    ax.bar_label(bars, fmt="%.0f")
    ax.set_ylabel("Mean absolute error (mm)"); ax.set_title("Rainfall imputation error by method")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "imputation_error_rainfall.png", dpi=150); plt.close(fig)


def main():
    config.OUT_DIR.mkdir(exist_ok=True)
    config.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 1. create the hypothetical data (same seed -> same data every time)
    raw, truth, meta = generate_data.generate()
    raw.to_csv(config.RAW_PATH, index=False)
    truth.to_csv(config.TRUTH_PATH, index=False)
    print(f"[1/5] Raw data created      : {len(raw)} rows")

    # 2. read it back exactly as text, so nothing is changed silently
    raw = pd.read_csv(config.RAW_PATH, dtype=str, keep_default_na=False)

    # 3. clean
    clean_df, log, snaps = cleaning.clean(raw)
    log_df = log.to_frame()
    log_df.to_csv(config.OUT_DIR / "cleaning_log.csv", index=False)
    clean_df.to_csv(config.CLEAN_PATH, index=False)
    print(f"[2/5] Cleaning finished     : {len(clean_df)} rows kept, {len(log_df)} actions logged")

    # 4. preprocess for modelling
    train, test, info = preprocessing.preprocess(clean_df)
    train.to_csv(config.TRAIN_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)
    print(f"[3/5] Preprocessing finished: train {len(train)} rows, test {len(test)} rows")

    # 5. evaluate against ground truth and make figures
    metrics = evaluate.evaluate(clean_df, snaps, truth, meta, log_df)
    metrics["preprocessing"] = info
    with open(config.OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print("[4/5] Evaluation saved      : outputs/metrics.json")
    make_figures(snaps, clean_df, metrics)
    print("[5/5] Figures saved         : outputs/figures/")

    print("\nKey results")
    print(f"  Duplicates injected / removed : {metrics['rows']['duplicates_injected']} / {metrics['rows']['duplicates_removed']}")
    for col, r in metrics["imputation"].items():
        print(f"  {col:<14} imputation MAE (used vs global mean): {r['mae_group_median_(used)']:.2f} vs {r['mae_global_mean']:.2f}")
    for name, r in metrics["outlier_strategies_yield"].items():
        print(f"  Yield, {name:<22}: RMSE {r['rmse']:.2f}, mean error {r['mean_error_pct']:.2f}%")


if __name__ == "__main__":
    main()
