"""
Phase 2 - Exploratory data analysis and physics validation of the ULA dataset.

What this does
--------------
1. Loads data/array_sweep_data.csv
2. Prints shape / describe() / null audit
3. Re-derives the grating-lobe label from the analytical formula and *asserts*
   zero mismatches (the dataset must be physically self-consistent)
4. Saves three figures to figures/
       gain_vs_spacing.png
       grating_lobe_distribution.png
       correlation_heatmap.png
5. Prints the grating_lobe class balance

Usage
-----
    python src/phase2_eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_style import (
    ACCENT,
    BG,
    DANGER,
    DARK_DIV,
    LABEL_SIZE,
    MUTED,
    PALETTE,
    PANEL,
    SAFE,
    TEXT,
    apply_dark_theme,
    save,
    style_axes,
    style_colorbar,
)

SEED = 42
np.random.seed(SEED)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "array_sweep_data.csv"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
apply_dark_theme()


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found - run `python src/phase1_generate_data.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"[Phase 2] Loaded {df.shape[0]} rows x {df.shape[1]} columns from {DATA_PATH.name}")
    return df


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    df = load_dataset()

    # ------------------------------------------------------------------ #
    # 1. Shape, describe, nulls
    # ------------------------------------------------------------------ #
    section("1. SHAPE / DTYPES / NULL AUDIT")
    print(f"shape            : {df.shape}")
    print("\ndtypes:")
    print(df.dtypes.to_string())
    print("\nnulls per column:")
    print(df.isnull().sum().to_string())

    section("2. DESCRIPTIVE STATISTICS")
    with pd.option_context("display.width", 140, "display.max_columns", 30):
        print(df.describe().T.to_string())

    # ------------------------------------------------------------------ #
    # 2. Physics validation of the label
    # ------------------------------------------------------------------ #
    section("3. GRATING-LOBE LABEL VERIFICATION (vs analytical physics)")
    expected_threshold = 1.0 / (1.0 + np.abs(np.sin(np.deg2rad(df["theta_scan_deg"]))))
    expected_label = (df["d_over_lambda"] > expected_threshold).astype(int)
    mismatches = int((expected_label != df["grating_lobe"]).sum())
    print("formula  : grating_lobe = 1 if d/lambda > 1 / (1 + |sin(theta)|) else 0")
    print(f"rows checked          : {len(df)}")
    print(f"mismatches            : {mismatches}")
    assert mismatches == 0, f"label/physics mismatch in {mismatches} rows"
    print("RESULT                : PASS - dataset label is physically consistent")

    # gl_margin must be exactly threshold - d/lambda
    margin_check = np.allclose(
        df["gl_margin"], (expected_threshold - df["d_over_lambda"]).to_numpy(), atol=1e-6
    )
    print(f"gl_margin recomputed  : {'PASS' if margin_check else 'FAIL'}")
    assert margin_check, "gl_margin does not equal gl_threshold - d_over_lambda"
    print(f"gl_margin sign agrees with label: "
          f"{'PASS' if ((df.gl_margin < 0).astype(int) == df.grating_lobe).all() else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # 3. Class balance
    # ------------------------------------------------------------------ #
    section("4. CLASS BALANCE - grating_lobe")
    counts = df["grating_lobe"].value_counts().sort_index()
    for label, count in counts.items():
        name = "1 (grating lobe)" if label == 1 else "0 (no grating lobe)"
        print(f"  {name:<22}: {count:>4} rows  ({100 * count / len(df):5.2f}%)")
    ratio = counts.max() / counts.min()
    print(f"  imbalance ratio      : {ratio:.2f}:1")

    # ------------------------------------------------------------------ #
    # 4. Figures
    # ------------------------------------------------------------------ #
    section("5. FIGURES")

    # --- (a) gain vs spacing, coloured by N ---------------------------- #
    fig, ax = plt.subplots(figsize=(12, 7))
    style_axes(ax)
    cmap = plt.get_cmap("viridis")
    ns = sorted(df["N"].unique())
    colours = {n: cmap(0.12 + 0.80 * i / (len(ns) - 1)) for i, n in enumerate(ns)}

    for n in ns:
        grp = df[df["N"] == n]
        ax.scatter(
            grp["d_over_lambda"], grp["gain_dBi"], s=95, alpha=0.9,
            color=colours[n], edgecolor=BG, linewidth=0.8,
            label=f"N = {n}   (mean {grp.gain_dBi.mean():.1f} dBi)", zorder=3,
        )
    xs = np.linspace(df.d_over_lambda.min(), df.d_over_lambda.max(), 200)
    for n in ns:
        ax.plot(xs, 10 * np.log10(n) + 2.15 - 0.5 * np.maximum(0, xs - 0.5),
                color=colours[n], lw=1.6, alpha=0.75, ls="--", zorder=2)

    ax.axvline(0.5, color=DANGER, ls=":", lw=2.0, zorder=1)
    ax.annotate(
        "d/$\\lambda$ = 0.5\nspacing penalty begins",
        xy=(0.5, ax.get_ylim()[1]),
        xytext=(0.55, ax.get_ylim()[1] - 0.55),
        color=DANGER, fontsize=LABEL_SIZE - 2, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec=DANGER, alpha=0.95),
    )
    ax.set_xlabel("Normalised element spacing   d/$\\lambda$")
    ax.set_ylabel("Gain  (dBi)")
    ax.set_title("Array gain vs. element spacing, coloured by number of elements")
    ax.legend(title="Elements", fontsize=11, title_fontsize=11,
              loc="center left", bbox_to_anchor=(0.02, 0.42))
    ax.margins(y=0.10)
    save(fig, FIG_DIR / "gain_vs_spacing.png")
    print("  saved -> figures/gain_vs_spacing.png")

    # --- (b) grating lobe distribution --------------------------------- #
    fig, ax = plt.subplots(figsize=(12, 7))
    style_axes(ax, grid_axis="y")
    counts_sorted = counts.reindex([0, 1]).fillna(0).astype(int)
    bars = ax.bar(
        ["0 \u2014 grating-lobe free", "1 \u2014 grating lobe present"],
        counts_sorted.values,
        color=[SAFE, DANGER], edgecolor="white", linewidth=1.2, width=0.56, zorder=3,
    )
    total = len(df)
    for bar, count in zip(bars, counts_sorted.values):
        ax.annotate(
            f"{count}\n({100 * count / total:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=LABEL_SIZE + 1,
            fontweight="bold", color=TEXT, linespacing=1.35,
        )
    ax.set_ylabel("Number of configurations")
    ax.set_xlabel("")
    ax.set_title("Grating-lobe class distribution  (250 ULA configurations)")
    ax.set_ylim(0, counts_sorted.max() * 1.24)
    save(fig, FIG_DIR / "grating_lobe_distribution.png")
    print("  saved -> figures/grating_lobe_distribution.png")

    # --- (c) correlation heatmap --------------------------------------- #
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12, 7))
    style_axes(ax, grid_axis="neither")
    ax.grid(False)
    heat = sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=DARK_DIV, vmin=-1, vmax=1,
        square=True, linewidths=1.0, linecolor=BG,
        cbar_kws={"shrink": 0.80}, ax=ax,
        annot_kws={"color": TEXT, "fontsize": 10, "fontweight": "bold"},
    )
    style_colorbar(heat.collections[0].colorbar)
    ax.set_title("Pearson correlation matrix of ULA sweep features")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", color=TEXT)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, FIG_DIR / "correlation_heatmap.png")
    print("  saved -> figures/correlation_heatmap.png")

    # correlation of each feature with the two targets, as text
    print("\n  |corr| with targets (top features):")
    for target in ("gain_dBi", "BW_MHz"):
        top = corr[target].drop(target).abs().sort_values(ascending=False).head(4)
        pretty = ", ".join(f"{k}={corr[target][k]:+.3f}" for k in top.index)
        print(f"    {target:<9}: {pretty}")

    print("\n[Phase 2] EDA complete.")


if __name__ == "__main__":
    main()
