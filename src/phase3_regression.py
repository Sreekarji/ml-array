"""
Phase 3 - Regression models for antenna-array gain and bandwidth prediction.

For BOTH targets (``gain_dBi`` and ``BW_MHz``) train nine regressors:

    1  Linear Regression
    2  Ridge Regression
    3  Decision Tree Regressor
    4  Random Forest Regressor      (n_estimators=100)
    5  Extra Trees Regressor        (n_estimators=100)
    6  Gradient Boosting Regressor
    7  XGBoost Regressor
    8  KNN Regressor                (k=5)
    9  Gaussian Process Regressor

Protocol
--------
features : [N, d_over_lambda, theta_scan_deg, log_N, gl_margin]
split    : 80/20 hold-out, random_state=42
scaling  : StandardScaler fitted on the TRAIN fold only (no leakage)
metrics  : R^2, variance score (explained_variance_score), MAE, MSE, RMSE

Outputs
-------
    results/regression_results.csv
    figures/model_comparison_gain.png
    figures/model_comparison_bw.png
    figures/predicted_vs_actual_gain.png
    figures/feature_importance.png

Usage
-----
    python src/phase3_regression.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "array_sweep_data.csv"
RESULT_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "figures"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["N", "d_over_lambda", "theta_scan_deg", "log_N", "gl_margin"]
TARGETS = ["gain_dBi", "BW_MHz"]
TARGET_UNITS = {"gain_dBi": "dBi", "BW_MHz": "MHz"}

plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
def build_models() -> dict[str, object]:
    """Fresh (unfitted) estimator for every model in the brief."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=SEED),
        "Decision Tree": DecisionTreeRegressor(random_state=SEED),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=SEED, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100, random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Gaussian Process": GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-2, 1e3))
            * RBF(length_scale=np.ones(len(FEATURES)), length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=SEED,
        ),
    }


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "Variance_Score": float(explained_variance_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    section("PHASE 3 - REGRESSION: GAIN AND BANDWIDTH PREDICTION")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found - run `python src/phase1_generate_data.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"[load] {df.shape[0]} rows x {df.shape[1]} columns <- {DATA_PATH.name}")
    print(f"[features] {FEATURES}")
    print(f"[targets ] {TARGETS}")

    X = df[FEATURES].to_numpy(dtype=float)

    records: list[dict] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}   # target -> model -> preds
    test_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted: dict[str, dict[str, object]] = {}            # target -> model -> fitted pipe

    for target in TARGETS:
        section(f"TARGET: {target}  ({TARGET_UNITS[target]})")
        y = df[target].to_numpy(dtype=float)

        # identical split for every model of this target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, shuffle=True
        )
        test_index[target] = (y_test, X_test)
        print(
            f"[split] train {X_train.shape[0]} / test {X_test.shape[0]} "
            f"(80/20, random_state={SEED})"
        )
        print(
            f"[stats] train mean {y_train.mean():.3f} std {y_train.std():.3f} | "
            f"test mean {y_test.mean():.3f} std {y_test.std():.3f}"
        )

        predictions[target] = {}
        fitted[target] = {}

        for name, estimator in build_models().items():
            # scaler fitted on the training fold only -> no leakage
            pipe = Pipeline(
                [("scaler", StandardScaler()), ("model", estimator)]
            )
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            elapsed = time.perf_counter() - t0

            scores = evaluate(y_test, y_pred)
            predictions[target][name] = y_pred
            fitted[target][name] = pipe
            records.append(
                {
                    "target": target,
                    "model": name,
                    **scores,
                    "fit_seconds": round(elapsed, 4),
                }
            )
            print(
                f"  {name:<20} R2={scores['R2']:+.4f}  Var={scores['Variance_Score']:+.4f}  "
                f"MAE={scores['MAE']:.4f}  MSE={scores['MSE']:.4f}  "
                f"RMSE={scores['RMSE']:.4f}   ({elapsed:6.2f}s)"
            )

    results = pd.DataFrame(records)[
        ["target", "model", "R2", "Variance_Score", "MAE", "MSE", "RMSE", "fit_seconds"]
    ]
    out_csv = RESULT_DIR / "regression_results.csv"
    results.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\n[results] {len(results)} rows -> {out_csv}")

    # ---- leaderboards ---------------------------------------------------- #
    section("LEADERBOARD")
    best: dict[str, str] = {}
    for target in TARGETS:
        sub = results[results.target == target].sort_values("R2", ascending=False)
        best[target] = sub.iloc[0]["model"]
        print(f"\n{target} - ranked by R^2")
        print(
            sub[["model", "R2", "Variance_Score", "MAE", "MSE", "RMSE"]]
            .to_string(index=False, float_format=lambda v: f"{v:9.4f}")
        )
    print("\n[best]" + "".join(f"  {t}: {m}" for t, m in best.items()))

    # ------------------------------------------------------------------ #
    # Figure: model comparison bar charts
    # ------------------------------------------------------------------ #
    order = results[results.target == TARGETS[0]].sort_values("R2", ascending=False)["model"]

    for target, fname, colour, label in [
        (TARGETS[0], "model_comparison_gain.png", "#2a6f97", "Gain"),
        (TARGETS[1], "model_comparison_bw.png", "#b5651d", "Bandwidth"),
    ]:
        sub = (
            results[results.target == target]
            .set_index("model")
            .loc[order]
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5.8))
        bars = ax.bar(sub["model"], sub["R2"], color=colour, alpha=0.88,
                      edgecolor="black", linewidth=0.6)
        # highlight the winner
        bars[0].set_color("#1b7f4b")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")
        # RMSE underneath so that models which are visually tied on R^2 are
        # still distinguishable (all scores cluster near 1.0 for gain)
        for bar, rmse in zip(bars, sub["RMSE"]):
            ax.annotate(
                f"RMSE {rmse:.3f}",
                (bar.get_x() + bar.get_width() / 2, 0.012),
                ha="center", va="bottom", fontsize=7.2, rotation=90,
                color="white", alpha=0.95,
            )

        ax.axhline(0, color="black", lw=0.9)
        threshold = 0.90 if target == "gain_dBi" else 0.70
        ax.axhline(threshold, color="crimson", ls="--", lw=1.3,
                   label=f"Phase 5 target: R² ≥ {threshold:.2f}")
        ax.set_ylabel("$R^2$  (test set)")
        ax.set_title(f"Model comparison — $R^2$ for {label.lower()} prediction ({target})")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["model"], rotation=32, ha="right")
        lo = min(0.0, float(sub["R2"].min())) - 0.08
        ax.set_ylim(lo, max(1.02, float(sub["R2"].max()) + 0.10))
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.35)
        fig.savefig(FIG_DIR / fname)
        plt.close(fig)
        print(f"[figure] figures/{fname}")

    # ------------------------------------------------------------------ #
    # Figure: predicted vs actual for the best GAIN model
    # ------------------------------------------------------------------ #
    y_test_gain, _ = test_index["gain_dBi"]
    best_gain = best["gain_dBi"]
    y_pred_best = predictions["gain_dBi"][best_gain]

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.scatter(y_test_gain, y_pred_best, s=58, alpha=0.82, c="#2a6f97",
               edgecolor="white", linewidth=0.7, label="test configurations")
    lo = min(y_test_gain.min(), y_pred_best.min()) - 0.6
    hi = max(y_test_gain.max(), y_pred_best.max()) + 0.6
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.6, label="$y = x$ (perfect prediction)")
    m = evaluate(y_test_gain, y_pred_best)
    ax.text(
        0.04, 0.96,
        f"best model: {best_gain}\n"
        f"$R^2$ = {m['R2']:.4f}\n"
        f"RMSE = {m['RMSE']:.3f} dBi\n"
        f"MAE  = {m['MAE']:.3f} dBi",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#2a6f97", alpha=0.92),
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Simulated (analytical) gain, dBi")
    ax.set_ylabel("Predicted gain, dBi")
    ax.set_title(f"Predicted vs. simulated gain — {best_gain}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "predicted_vs_actual_gain.png")
    plt.close(fig)
    print("[figure] figures/predicted_vs_actual_gain.png")

    # ------------------------------------------------------------------ #
    # Figure: Random Forest feature importance (both targets side by side)
    # ------------------------------------------------------------------ #
    rf_fi: dict[str, np.ndarray] = {}
    for target in TARGETS:
        rf_fi[target] = fitted[target]["Random Forest"].named_steps[
            "model"
        ].feature_importances_

    fi_df = pd.DataFrame(rf_fi, index=FEATURES)
    fi_df = fi_df.loc[fi_df.mean(axis=1).sort_values(ascending=False).index]
    print("\n[importance] Random Forest feature_importances_")
    print(fi_df.to_string(float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    xpos = np.arange(len(fi_df))
    width = 0.4
    ax.bar(xpos - width / 2, fi_df["gain_dBi"], width, label="gain_dBi",
           color="#2a6f97", edgecolor="black", linewidth=0.5)
    ax.bar(xpos + width / 2, fi_df["BW_MHz"], width, label="BW_MHz",
           color="#b5651d", edgecolor="black", linewidth=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(fi_df.index)
    ax.set_ylabel("Feature importance (mean decrease in impurity)")
    ax.set_title("Random Forest feature importance for gain and bandwidth")
    ax.legend()
    ax.grid(axis="y", alpha=0.35)
    fig.savefig(FIG_DIR / "feature_importance.png")
    plt.close(fig)
    print("[figure] figures/feature_importance.png")

    fi_df.to_csv(RESULT_DIR / "feature_importance.csv", float_format="%.6f")

    section("PHASE 3 COMPLETE")
    print(f"  results  : {out_csv.relative_to(REPO_ROOT)}  ({len(results)} rows)")
    print("  figures  : model_comparison_gain.png, model_comparison_bw.png,")
    print("             predicted_vs_actual_gain.png, feature_importance.png")


if __name__ == "__main__":
    main()
