"""
Shared dark theme for every figure in the project.

Single source of truth so that all phase scripts (and the Phase 2 notebook)
render identical styling. Import as::

    from plot_style import PALETTE, apply_dark_theme, style_axes, save

Design
------
    canvas    #0f172a   (dark navy)
    axes      #1e293b   (slate panel)
    text      white
    accent    #38bdf8   (sky blue)
    grid      white, alpha 0.15, dashed
    figures   12 x 7 in at 200 dpi
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe, must precede pyplot import

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
BG = "#0f172a"
PANEL = "#1e293b"
TEXT = "#ffffff"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"

PALETTE = [
    "#38bdf8",  # sky
    "#f472b6",  # pink
    "#34d399",  # emerald
    "#fb923c",  # orange
    "#a78bfa",  # violet
    "#facc15",  # yellow
    "#f87171",  # red
    "#4ade80",  # green
    "#60a5fa",  # blue
]

# semantic colours used across figures
SAFE = "#34d399"     # no grating lobe
DANGER = "#f87171"   # grating lobe present
THRESHOLD = "#f87171"

FIGSIZE = (12, 7)
DPI = 200
LABEL_SIZE = 13
TICK_SIZE = 13
TITLE_SIZE = 15
LEGEND_SIZE = 11

# a diverging map that stays legible on a dark canvas:
# blue (-1) -> panel dark (0) -> red (+1)
DARK_DIV = LinearSegmentedColormap.from_list(
    "dark_div", ["#60a5fa", "#1e293b", "#f87171"]
)


def apply_dark_theme() -> None:
    """Set global rcParams for the dark theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "figure.edgecolor": BG,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.facecolor": PANEL,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "axes.grid": True,
            "grid.color": "white",
            "grid.alpha": 0.15,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "text.color": TEXT,
            "legend.facecolor": PANEL,
            "legend.edgecolor": MUTED,
            "legend.labelcolor": TEXT,
            "legend.fontsize": LEGEND_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "font.family": "sans-serif",
        }
    )


def style_axes(ax, grid_axis: str = "both") -> None:
    """Apply the per-axes parts of the theme (spines, grid, ticks).

    ``grid_axis`` accepts "x", "y", "both" or None/"none" to disable the grid.
    """
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=TEXT, labelsize=TICK_SIZE)
    if grid_axis in (None, "none", "neither"):
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, color="white", alpha=0.15,
                linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)


def style_colorbar(cbar) -> None:
    """Dark-theme a colorbar (ticks, outline, label)."""
    cbar.ax.tick_params(colors=TEXT, labelsize=TICK_SIZE - 1)
    cbar.outline.set_edgecolor(MUTED)
    cbar.ax.yaxis.label.set_color(TEXT)
    cbar.ax.yaxis.label.set_size(LABEL_SIZE)


def save(fig, path: Path, tight: bool = True) -> Path:
    """Save with the standard high-DPI settings and close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG, edgecolor=BG)
    plt.close(fig)
    return path
