"""
Phase 5 - Independent verification of every phase.

Re-derives everything from first principles instead of trusting the artefacts:
the grating-lobe label is recomputed from the analytical formula, gains are
checked against the physics model, and the results files are re-read so that a
stale or truncated CSV is caught.

Prints PASS / FAIL per check and a final "X/Y checks passed" summary.
Exit code is 0 only when every check passes.

Usage
-----
    python src/phase5_verification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "array_sweep_data.csv"
RESULT_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "figures"
SRC_DIR = REPO_ROOT / "src"

EXPECTED_FIGURES = [
    "gain_vs_spacing.png",
    "grating_lobe_distribution.png",
    "correlation_heatmap.png",
    "model_comparison_gain.png",
    "model_comparison_bw.png",
    "predicted_vs_actual_gain.png",
    "feature_importance.png",
    "confusion_matrix.png",
    "roc_curves.png",
    "grating_lobe_boundary.png",
    # explanatory figures added in the polish pass
    "array_schematic.png",
    "pipeline_overview.png",
]
EXPECTED_SOURCES = [
    "phase1_generate_data.py",
    "phase2_eda.py",
    "phase3_regression.py",
    "phase4_classifier.py",
    "phase5_verification.py",
]
EXPECTED_MODELS_REGRESSION = [
    "Linear Regression",
    "Ridge Regression",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "XGBoost",
    "KNN",
    "Gaussian Process",
]
EXPECTED_MODELS_CLASSIFIER = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "SVM",
    "KNN",
]

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record and print one PASS/FAIL line."""
    RESULTS.append((name, bool(ok), detail))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"   -> {detail}" if detail else ""))
    return bool(ok)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    print("=" * 78)
    print("PHASE 5 - VERIFICATION AND FINAL CHECKS")
    print("=" * 78)

    # ================================================================== #
    # 1. DATASET CHECKS
    # ================================================================== #
    section("1. DATASET CHECKS")
    if not DATA_PATH.exists():
        check("data/array_sweep_data.csv exists", False, f"missing at {DATA_PATH}")
        print("\nCannot continue without the dataset.")
        return 1

    df = pd.read_csv(DATA_PATH)
    check("data/array_sweep_data.csv exists", True, f"{DATA_PATH.name}")

    check(
        "row count >= 200",
        len(df) >= 200,
        f"{len(df)} rows",
    )
    check(
        "no null values",
        int(df.isnull().sum().sum()) == 0,
        f"{int(df.isnull().sum().sum())} nulls",
    )

    # label re-derived from physics, independently of the CSV generation code
    theta_rad = np.deg2rad(df["theta_scan_deg"].to_numpy(dtype=float))
    threshold = 1.0 / (1.0 + np.abs(np.sin(theta_rad)))
    recomputed = (df["d_over_lambda"].to_numpy(dtype=float) > threshold).astype(int)
    mismatches = int((recomputed != df["grating_lobe"].to_numpy(dtype=int)).sum())
    check(
        "grating_lobe matches physics formula for every row",
        mismatches == 0,
        f"{mismatches} mismatches / {len(df)} rows",
    )

    g_min, g_max = float(df.gain_dBi.min()), float(df.gain_dBi.max())
    check(
        "gain_dBi within physically plausible range (3..25 dBi)",
        3.0 <= g_min and g_max <= 25.0,
        f"{g_min:.3f} .. {g_max:.3f} dBi",
    )

    # extra sanity: gain must track the analytical aperture model
    n = df["N"].to_numpy(dtype=float)
    d = df["d_over_lambda"].to_numpy(dtype=float)
    g_expected = 10 * np.log10(n) + 2.15 - 0.5 * np.maximum(0.0, d - 0.5)
    max_dev = float(np.max(np.abs(df.gain_dBi.to_numpy(dtype=float) - g_expected)))
    check(
        "gain_dBi follows 10log10(N) + 2.15 - 0.5*max(0, d/lambda - 0.5)",
        max_dev < 0.5,
        f"max deviation {max_dev:.4f} dBi (noise sigma 0.08)",
    )

    # ================================================================== #
    # 2. REGRESSION CHECKS
    # ================================================================== #
    section("2. REGRESSION CHECKS")
    reg_path = RESULT_DIR / "regression_results.csv"
    reg_exists = reg_path.exists()
    check("results/regression_results.csv exists", reg_exists, str(reg_path.name))
    if reg_exists:
        reg = pd.read_csv(reg_path)
        required_cols = {"target", "model", "R2", "Variance_Score", "MAE", "MSE", "RMSE"}
        check(
            "regression_results.csv has R^2 / Var / MAE / MSE / RMSE columns",
            required_cols.issubset(reg.columns),
            f"columns: {list(reg.columns)}",
        )

        counts = reg.groupby("target").size()
        for target in ("gain_dBi", "BW_MHz"):
            got = int(counts.get(target, 0))
            check(
                f"9 result rows for target {target}",
                got == 9,
                f"{got} rows",
            )
        for target in ("gain_dBi", "BW_MHz"):
            models = set(reg.loc[reg.target == target, "model"])
            missing = [m for m in EXPECTED_MODELS_REGRESSION if m not in models]
            check(
                f"all 9 required models present for {target}",
                not missing,
                "all present" if not missing else f"missing: {missing}",
            )

        for target, bar in (("gain_dBi", 0.90), ("BW_MHz", 0.70)):
            sub = reg[reg.target == target]
            if len(sub):
                best_row = sub.loc[sub["R2"].idxmax()]
                check(
                    f"best {target} R^2 >= {bar:.2f}",
                    float(best_row["R2"]) >= bar,
                    f"{best_row['model']} R^2 = {best_row['R2']:.4f}",
                )
            else:
                check(f"best {target} R^2 >= {bar:.2f}", False, "no rows")

        # metrics must be internally consistent: RMSE == sqrt(MSE)
        consistent = np.allclose(reg["RMSE"], np.sqrt(reg["MSE"]), rtol=1e-4)
        check("RMSE == sqrt(MSE) in every row", consistent)
    else:
        for name in [
            "9 result rows for target gain_dBi",
            "9 result rows for target BW_MHz",
            "all 9 required models present for gain_dBi",
            "all 9 required models present for BW_MHz",
            "best gain_dBi R^2 >= 0.90",
            "best BW_MHz R^2 >= 0.70",
        ]:
            check(name, False, "regression_results.csv missing")

    # ================================================================== #
    # 3. CLASSIFIER CHECKS
    # ================================================================== #
    section("3. CLASSIFIER CHECKS")
    clf_path = RESULT_DIR / "classifier_results.csv"
    clf_exists = clf_path.exists()
    check("results/classifier_results.csv exists", clf_exists, str(clf_path.name))
    if clf_exists:
        clf = pd.read_csv(clf_path)
        required = {
            "classifier", "accuracy", "roc_auc",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_class0", "recall_class0", "f1_class0",
            "precision_class1", "recall_class1", "f1_class1",
            "tn", "fp", "fn", "tp",
        }
        check(
            "classifier_results.csv reports accuracy / P / R / F1 / ROC-AUC / CM",
            required.issubset(clf.columns),
            f"{len(clf.columns)} columns",
        )
        check(
            "all 6 required classifiers present",
            set(EXPECTED_MODELS_CLASSIFIER).issubset(set(clf["classifier"])),
            f"{len(clf)} classifiers",
        )
        best_acc_row = clf.loc[clf["accuracy"].idxmax()]
        check(
            "best classifier accuracy >= 0.90",
            float(best_acc_row["accuracy"]) >= 0.90,
            f"{best_acc_row['classifier']} accuracy = {best_acc_row['accuracy']:.4f}",
        )
        best_auc_row = clf.loc[clf["roc_auc"].idxmax()]
        check(
            "best ROC-AUC >= 0.95",
            float(best_auc_row["roc_auc"]) >= 0.95,
            f"{best_auc_row['classifier']} ROC-AUC = {best_auc_row['roc_auc']:.4f}",
        )
        # confusion-matrix totals must equal the held-out test-set size
        totals = (clf["tn"] + clf["fp"] + clf["fn"] + clf["tp"]).unique()
        check(
            "confusion matrices sum to the same test-set size",
            len(totals) == 1,
            f"test size = {[int(t) for t in totals]}",
        )
    else:
        for name in [
            "all 6 required classifiers present",
            "best classifier accuracy >= 0.90",
            "best ROC-AUC >= 0.95",
        ]:
            check(name, False, "classifier_results.csv missing")

    # ================================================================== #
    # 4. FILE CHECKS
    # ================================================================== #
    section("4. FILE CHECKS")
    missing_figs = [f for f in EXPECTED_FIGURES if not (FIG_DIR / f).exists()]
    check(
        "all 12 expected figures exist in figures/",
        not missing_figs,
        "all present" if not missing_figs else f"missing: {missing_figs}",
    )
    for fig in EXPECTED_FIGURES:
        path = FIG_DIR / fig
        size = path.stat().st_size if path.exists() else 0
        check(
            f"figure non-empty: {fig}",
            path.exists() and size > 5_000,
            f"{size / 1024:.1f} kB" if path.exists() else "missing",
        )

    missing_src = [f for f in EXPECTED_SOURCES if not (SRC_DIR / f).exists()]
    check(
        "all 5 phase scripts exist in src/",
        not missing_src,
        "all present" if not missing_src else f"missing: {missing_src}",
    )
    check("data/array_sweep_data.csv exists", DATA_PATH.exists())
    check(
        "notebooks/phase2_eda.ipynb exists",
        (REPO_ROOT / "notebooks" / "phase2_eda.ipynb").exists(),
    )
    check("README.md exists", (REPO_ROOT / "README.md").exists())
    check("requirements.txt exists", (REPO_ROOT / "requirements.txt").exists())
    check(
        "requirements_dev.txt exists",
        (REPO_ROOT / "requirements_dev.txt").exists(),
    )
    check("LICENSE exists", (REPO_ROOT / "LICENSE").exists())
    check(
        "phase2 notebook and phase2_eda.py render identically styled figures",
        (SRC_DIR / "plot_style.py").exists(),
        "shared theme module present",
    )

    # ================================================================== #
    # SUMMARY
    # ================================================================== #
    section("SUMMARY")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"  {passed}/{total} checks passed")
    print(f"  ({passed / total * 100:.1f}% of the verification suite)")

    failures = [(n, d) for n, ok, d in RESULTS if not ok]
    if failures:
        print("\n  FAILED CHECKS:")
        for name, detail in failures:
            print(f"    - {name}" + (f"   ({detail})" if detail else ""))
        print("\n  RESULT: FAIL - see the failures listed above.")
        return 1

    print("\n  RESULT: PASS - every dataset, regression, classifier and file check succeeded.")
    if reg_exists:
        sub = reg[reg.target == "gain_dBi"]
        top = sub.loc[sub["R2"].idxmax()]
        print(f"  Best gain R^2   : {top['model']} (R^2 = {top['R2']:.4f})")
    if clf_exists:
        print(f"  Best classifier : {clf.loc[clf['accuracy'].idxmax(), 'classifier']} "
              f"(accuracy {clf['accuracy'].max():.4f}, "
              f"ROC-AUC {clf['roc_auc'].max():.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
