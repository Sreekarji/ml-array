"""
Two extra explanatory figures (dark theme, no external assets):

    figures/array_schematic.png    - annotated ULA geometry: N=8 elements,
                                     spacing d, scan angle, main beam and
                                     a spurious grating lobe
    figures/pipeline_overview.png  - end-to-end ML pipeline flowchart

Both are pure matplotlib primitives so they regenerate anywhere.

Usage
-----
    python src/figures_extra.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (  # noqa: E402
    ACCENT,
    BG,
    DANGER,
    FIGSIZE,
    MUTED,
    PALETTE,
    PANEL,
    SAFE,
    TEXT,
    apply_dark_theme,
    save,
    style_axes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"


# --------------------------------------------------------------------------- #
# 1. ULA schematic
# --------------------------------------------------------------------------- #
def array_schematic() -> Path:
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    style_axes(ax, grid_axis="both")
    # Fixed data window wide enough for both beams AND the left-hand label.
    # adjustable="box" keeps these limits exact while preserving round elements.
    ax.set_xlim(-0.58, 1.32)
    ax.set_ylim(-0.30, 1.10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    # --- elements ------------------------------------------------------- #
    n_el = 8
    x = np.linspace(0.0, 1.0, n_el)
    y = np.zeros_like(x)
    ax.axhline(0, color=MUTED, lw=1.4, zorder=1, alpha=0.8)
    ax.scatter(x, y, s=420, c=PALETTE[0], edgecolor="white", linewidth=1.6, zorder=3)
    ax.scatter(x, y, s=90, c=BG, edgecolor="none", zorder=4, alpha=0.35)
    for i, (xi, yi) in enumerate(zip(x, y), start=1):
        ax.annotate(
            str(i), (xi, yi), ha="center", va="center",
            fontsize=10, fontweight="bold", color=BG, zorder=6,
        )

    # --- spacing annotation between first two elements ------------------ #
    xa, xb = x[0], x[1]
    y_arr = -0.105
    ax.annotate(
        "", xy=(xb, y_arr), xytext=(xa, y_arr),
        arrowprops=dict(arrowstyle="<|-|>", color=ACCENT, lw=1.8,
                        shrinkA=0, shrinkB=0, mutation_scale=16),
    )
    ax.plot([xa, xa], [0, y_arr], color=MUTED, lw=0.9, ls=":", alpha=0.9)
    ax.plot([xb, xb], [0, y_arr], color=MUTED, lw=0.9, ls=":", alpha=0.9)
    ax.text(
        (xa + xb) / 2, y_arr - 0.055, "d", fontsize=18, fontweight="bold",
        color=ACCENT, ha="center", va="top",
    )

    # --- broadside reference ------------------------------------------- #
    ax.annotate(
        "", xy=(x[3], 0.88), xytext=(x[3], 0.06),
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2, ls=":"),
    )
    ax.text(x[3] + 0.025, 0.90, "broadside", fontsize=11, color=MUTED, style="italic")

    # --- main beam (scan angle theta from broadside) -------------------- #
    theta0 = 34.0                     # degrees off broadside
    rad = np.deg2rad(theta0)
    origin = np.array([x[3], 0.06])
    length = 0.80
    tip = origin + length * np.array([np.sin(rad), np.cos(rad)])
    ax.annotate(
        "", xy=tuple(tip), xytext=tuple(origin),
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=3.0, mutation_scale=22),
    )
    ax.text(
        tip[0] + 0.015, tip[1] + 0.02, "Main beam", fontsize=13,
        fontweight="bold", color=ACCENT, ha="left", va="bottom",
    )

    # angle arc between broadside and main beam
    r_arc = 0.26
    arc = Arc(
        (x[3], 0.06), 2 * r_arc, 2 * r_arc, angle=0,
        theta1=90 - theta0, theta2=90, color=TEXT, lw=1.4,
    )
    ax.add_patch(arc)
    ax.text(
        x[3] + 0.012, 0.06 + r_arc * 1.22, r"$\theta_0$",
        fontsize=17, color=TEXT, ha="left", va="center",
    )

    # --- grating lobe (fainter, dashed, mirrored) ----------------------- #
    theta_gl = -46.0
    rad_gl = np.deg2rad(theta_gl)
    tip_gl = origin + 0.62 * np.array([np.sin(rad_gl), np.cos(rad_gl)])
    ax.annotate(
        "", xy=tuple(tip_gl), xytext=tuple(origin),
        arrowprops=dict(arrowstyle="-|>", color=DANGER, lw=2.2, ls="--",
                        mutation_scale=18, alpha=0.85),
    )
    ax.text(
        tip_gl[0] - 0.03, tip_gl[1], "Grating lobe\n(spurious)",
        fontsize=12, color=DANGER, ha="right", va="center", alpha=0.95,
    )

    # --- title / subtitle ---------------------------------------------- #
    ax.set_title(
        "Uniform Linear Array (N = 8)", fontsize=20, fontweight="bold",
        color=TEXT, pad=16,
    )
    ax.text(
        0.5, -0.275,
        "Elements spaced d apart.  When d/\u03bb exceeds 1/(1+|sin \u03b8\u2080|) "
        "a second main lobe enters visible space.",
        ha="center", va="bottom", transform=ax.transAxes,
        fontsize=12, color=MUTED, style="italic",
    )

    out = save(fig, FIG_DIR / "array_schematic.png")
    print(f"  saved -> figures/{out.name}")
    return out


# --------------------------------------------------------------------------- #
# 2. Pipeline overview
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, label, color, fontsize=12, text_color=None):
    """Rounded box with centred multi-line label."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.030",
        linewidth=1.8, edgecolor=color, facecolor=PANEL, zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x, y, label, ha="center", va="center", fontsize=fontsize,
        color=text_color or TEXT, fontweight="bold", zorder=4, linespacing=1.35,
    )
    return box


def _arrow(ax, p0, p1, color=MUTED, lw=2.0, style="-|>", ls="-", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, color=color, lw=lw, ls=ls,
            mutation_scale=18, shrinkA=6, shrinkB=6, zorder=2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def pipeline_overview() -> Path:
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.set_facecolor(PANEL)

    IN_C, PROC_C, OUT_C = PALETTE[0], PALETTE[4], PALETTE[2]   # sky / violet / emerald

    # ---- lane labels ---- #
    for y, txt, c in [
        (6.72, "INPUT", IN_C),
        (4.20, "PROCESSING", PROC_C),
        (1.55, "PREDICTIONS", OUT_C),
    ]:
        ax.text(0.30, y, txt, fontsize=11, fontweight="bold", color=c,
                ha="left", va="center", alpha=0.95)

    # ---- row 1: input -> generator -> csv ---- #
    y1, h1 = 6.15, 1.05
    _box(ax, 1.95, y1, 2.75, h1, "ULA Parameters\n(N, d/\u03bb, \u03b8)", IN_C, 13)
    _box(ax, 5.55, y1, 2.55, h1, "Dataset\nGenerator", PROC_C, 13)
    _box(ax, 9.35, y1, 2.55, h1, "250-row CSV\narray_sweep_data.csv", IN_C, 12)

    _arrow(ax, (3.33, y1), (4.27, y1), ACCENT)
    _arrow(ax, (6.83, y1), (8.07, y1), ACCENT)

    # ---- row 2: feature engineering -> ML models ---- #
    y2, h2 = 4.15, 1.15
    _box(ax, 3.75, y2, 3.10, h2,
         "Feature Engineering\nlog_N  \u00b7  gl_margin", PROC_C, 12)
    _box(ax, 9.30, y2, 3.55, h2,
         "ML Models\n9 regressors  \u00b7  6 classifiers", PROC_C, 11.5)

    # csv -> feature engineering (curved, from below the CSV box)
    _arrow(ax, (9.35, y1 - h1 / 2 - 0.03), (4.60, y2 + h2 / 2 + 0.03),
           ACCENT, ls="--", rad=0.18)
    _arrow(ax, (5.32, y2), (7.62, y2), ACCENT)

    # ---- row 3: three prediction outputs ---- #
    y3, h3 = 1.50, 1.12
    outs = [
        (2.10, "Gain Prediction\ngain_dBi   \u00b7   R\u00b2 0.9997"),
        (6.70, "BW Prediction\nBW_MHz   \u00b7   R\u00b2 0.8075"),
        (11.15, "Grating Lobe\nClassification   \u00b7   100%"),
    ]
    for xo, label in outs:
        _box(ax, xo, y3, 3.55, h3, label, OUT_C, 11.5)
        _arrow(ax, (9.05, y2 - h2 / 2 - 0.03), (xo, y3 + h3 / 2 + 0.03),
               OUT_C, rad=0.0 if xo > 8 else 0.12)

    # ---- verification footer ---- #
    ax.text(
        6.70, 0.36,
        "Phase 5 verification: 42/42 checks passed  \u00b7  all seeds = 42",
        ha="center", va="center", fontsize=12, color=SAFE, fontweight="bold",
    )

    ax.set_title(
        "End-to-End ML Pipeline", fontsize=20, fontweight="bold", color=TEXT, pad=14,
    )
    out = save(fig, FIG_DIR / "pipeline_overview.png")
    print(f"  saved -> figures/{out.name}")
    return out


if __name__ == "__main__":
    print("[figures_extra] building explanatory figures ...")
    array_schematic()
    pipeline_overview()
    print("[figures_extra] done.")
