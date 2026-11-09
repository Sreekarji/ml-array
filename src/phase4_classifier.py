"""
Phase 4 - Grating-lobe classification.

Binary task: given the array geometry (N, d/lambda, theta_scan) plus the
engineered margins, predict whether the configuration will throw a grating
lobe into visible space.

Six classifiers, all with probability output so ROC curves can be drawn:

    1  Logistic Regression      (class_weight='balanced')
    2  Random Forest            (n_estimators=100, class_weight='balanced')
    3  Gradient Boosting
    4  XGBoost
    5  SVM                      (probability=True, class_weight='balanced')
    6  KNN                      (k=5)

Protocol
--------
stratified 80/20 split, random_state=42, features scaled with a StandardScaler
fitted on the training fold only.

Metrics
-------
accuracy, precision / recall / F1 (macro AND per class), ROC-AUC,
confusion matrix (written to CSV + printed).

Outputs
-------
    results/classifier_results.csv
    results/confusion_matrices.csv
    figures/confusion_matrix.png
    figures/roc_curves.png
    figures/grating_lobe_boundary.png

Usage
-----
    python src/phase4_classifier.py
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
TARGET = "grating_lobe"

# distinct, print-safe colours for the six ROC curves
# three classifiers score a perfect AUC, so wide styles + dashes are needed for
# every curve to remain visible where they coincide
LINESTYLES = {
    "Logistic Regression": (0, (6, 2)),
    "Random Forest": "solid",
    "Gradient Boosting": (0, (1, 1.4)),
    "XGBoost": (0, (5, 1.5, 1, 1.5)),
    "SVM": (0, (3, 1.5)),
    "KNN": (0, (8, 2, 1, 2)),
}
LINEWIDTHS = {
    "Logistic Regression": 2.0,
    "Random Forest": 7.0,
    "Gradient Boosting": 4.2,
    "XGBoost": 3.0,
    "SVM": 2.2,
    "KNN": 2.0,
}

COLOURS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest": "#2ca02c",
    "Gradient Boosting": "#d62728",
    "XGBoost": "#9467bd",
    "SVM": "#ff7f0e",
    "KNN": "#17becf",
}

plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"


def build_classifiers() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=SEED,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    section("PHASE 4 - GRATING-LOBE CLASSIFIER")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found - run `python src/phase1_generate_data.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"[load] {df.shape[0]} rows x {df.shape[1]} columns <- {DATA_PATH.name}")
    print(f"[features] {FEATURES}")
    print(f"[target  ] {TARGET}  (1 = grating lobe present)")

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\n[split] STRATIFIED 80/20, random_state={SEED}")
    print(f"  train {X_train.shape[0]:>3} rows  | class counts {np.bincount(y_train)}")
    print(f"  test  {X_test.shape[0]:>3} rows  | class counts {np.bincount(y_test)}")

    records: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    proba: dict[str, np.ndarray] = {}
    cms: dict[str, np.ndarray] = {}

    for name, clf in build_classifiers().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)[:, 1]
        elapsed = time.perf_counter() - t0

        fitted[name] = pipe
        proba[name] = y_score
        cms[name] = confusion_matrix(y_test, y_pred)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)
        p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        p_w, r_w, f_w, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        p_c, r_c, f_c, sup_c = precision_recall_fscore_support(
            y_test, y_pred, labels=[0, 1], zero_division=0
        )

        records.append(
            {
                "classifier": name,
                "accuracy": acc,
                "roc_auc": auc,
                "precision_macro": p_mac,
                "recall_macro": r_mac,
                "f1_macro": f_mac,
                "precision_weighted": p_w,
                "recall_weighted": r_w,
                "f1_weighted": f_w,
                "precision_class0": p_c[0],
                "recall_class0": r_c[0],
                "f1_class0": f_c[0],
                "support_class0": int(sup_c[0]),
                "precision_class1": p_c[1],
                "recall_class1": r_c[1],
                "f1_class1": f_c[1],
                "support_class1": int(sup_c[1]),
                "tn": int(cms[name][0, 0]),
                "fp": int(cms[name][0, 1]),
                "fn": int(cms[name][1, 0]),
                "tp": int(cms[name][1, 1]),
                "fit_seconds": round(elapsed, 4),
            }
        )

        print(f"\n  {name}")
        print(f"    accuracy          : {acc:.4f}")
        print(f"    ROC-AUC           : {auc:.4f}")
        print(
            f"    macro  P/R/F1     : {p_mac:.4f} / {r_mac:.4f} / {f_mac:.4f}"
        )
        print(
            f"    weighted P/R/F1   : {p_w:.4f} / {r_w:.4f} / {f_w:.4f}"
        )
        print("    per-class:")
        for idx, cls in enumerate(["0 (no lobe)", "1 (lobe)"]):
            print(
                f"      {cls:<12} P={p_c[idx]:.4f}  R={r_c[idx]:.4f}  "
                f"F1={f_c[idx]:.4f}  n={sup_c[idx]}"
            )
        print("    confusion matrix (rows=true, cols=pred):")
        for row in cms[name]:
            print(f"      {row}")
        print(f"    fit time          : {elapsed:.2f}s")

    # rank by accuracy, then ROC-AUC, then macro-F1 (deterministic tie-break so
    # the "best classifier" chosen for the figures never depends on row order)
    results = pd.DataFrame(records).sort_values(
        ["accuracy", "roc_auc", "f1_macro"], ascending=False
    ).reset_index(drop=True)
    out_csv = RESULT_DIR / "classifier_results.csv"
    results.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\n[results] {len(results)} rows -> {out_csv}")

    # confusion matrices also dumped as a tidy CSV
    cm_rows = []
    for name, cm in cms.items():
        for true_label in (0, 1):
            for pred_label in (0, 1):
                cm_rows.append(
                    {
                        "classifier": name,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(cm[true_label, pred_label]),
                    }
                )
    pd.DataFrame(cm_rows).to_csv(
        RESULT_DIR / "confusion_matrices.csv", index=False
    )

    section("LEADERBOARD - ranked by accuracy")
    print(
        results[
            ["classifier", "accuracy", "roc_auc", "precision_macro",
             "recall_macro", "f1_macro"]
        ].to_string(index=False, float_format=lambda v: f"{v:9.4f}")
    )
    best = results.iloc[0]["classifier"]
    print(f"\n[best] classifier: {best}  "
          f"(accuracy {results.iloc[0]['accuracy']:.4f}, "
          f"ROC-AUC {results.iloc[0]['roc_auc']:.4f})")

    # ------------------------------------------------------------------ #
    # Figure: confusion matrix of the best classifier
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cms[best], display_labels=["no lobe (0)", "lobe (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(
        f"Confusion matrix — best classifier: {best}\n"
        f"accuracy {results.iloc[0]['accuracy']:.3f} | "
        f"ROC-AUC {results.iloc[0]['roc_auc']:.3f} | "
        f"n = {len(y_test)} held-out configurations",
        fontsize=11,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.savefig(FIG_DIR / "confusion_matrix.png")
    plt.close(fig)
    print("[figure] figures/confusion_matrix.png")

    # ------------------------------------------------------------------ #
    # Figure: ROC curves, all classifiers on one axis
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    for name in build_classifiers():
        fpr, tpr, _ = roc_curve(y_test, proba[name])
        auc = roc_auc_score(y_test, proba[name])
        ax.plot(
            fpr, tpr, lw=LINEWIDTHS[name], color=COLOURS[name],
            ls=LINESTYLES[name], alpha=0.9,
            label=f"{name}  (AUC = {auc:.4f})",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="chance (AUC = 0.5)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — grating-lobe classifiers (held-out test set)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "roc_curves.png")
    plt.close(fig)
    print("[figure] figures/roc_curves.png")

    # ------------------------------------------------------------------ #
    # Figure: grating-lobe boundary in the (d/lambda, theta) plane
    # ------------------------------------------------------------------ #
    theta_grid = np.linspace(0, 75, 400)
    boundary = 1.0 / (1.0 + np.abs(np.sin(np.deg2rad(theta_grid))))

    fig, ax = plt.subplots(figsize=(9.2, 6.2))

    # background: decision surface of the best tree/ensemble model evaluated on
    # (d/lambda, theta) with N fixed at its median and the derived features
    # recomputed consistently.
    n_ref = int(df["N"].median())
    d_axis = np.linspace(df.d_over_lambda.min() - 0.05, df.d_over_lambda.max() + 0.05, 260)
    t_axis = np.linspace(0, 75, 260)
    DD, TT = np.meshgrid(d_axis, t_axis)
    thr = 1.0 / (1.0 + np.abs(np.sin(np.deg2rad(TT))))
    grid = np.column_stack(
        [
            np.full(DD.size, n_ref),
            DD.ravel(),
            TT.ravel(),
            np.full(DD.size, np.log2(n_ref)),
            (thr - DD).ravel(),
        ]
    )
    Z = fitted[best].predict_proba(grid)[:, 1].reshape(DD.shape)
    # shaded decision surface (P(grating lobe)) behind the measured labels
    contour = ax.contourf(DD, TT, Z, levels=20, cmap="RdYlGn_r", alpha=0.30)
    ax.contour(DD, TT, Z, levels=[0.5], colors="black", linewidths=1.4,
               linestyles="-", alpha=0.8)
    cbar = fig.colorbar(contour, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(f"P(grating lobe) — {best} decision surface", fontsize=9)

    scatter = ax.scatter(
        df["d_over_lambda"],
        df["theta_scan_deg"],
        c=df["grating_lobe"],
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        s=52,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.9,
    )
    ax.plot(
        boundary, theta_grid, "k--", lw=2.4,
        label=r"theory: $d/\lambda = 1/(1+|\sin\theta_0|)$",
    )
    ax.set_xlabel(r"Normalised element spacing  $d/\lambda$")
    ax.set_ylabel(r"Scan angle  $\theta_0$  (degrees)")
    ax.set_title(
        "Grating-lobe boundary — measured labels, analytical curve and\n"
        f"decision surface of the best classifier ({best}, N = {n_ref})"
    )
    ax.set_xlim(d_axis.min(), d_axis.max())
    ax.set_ylim(0, 75)
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc="#2ca02c", mec="black",
                   ms=9, label="label 0 — grating-lobe free"),
        plt.Line2D([], [], marker="o", ls="", mfc="#d62728", mec="black",
                   ms=9, label="label 1 — grating lobe present"),
        plt.Line2D([], [], color="black", ls="--", lw=2.2,
                   label=r"theory: $d/\lambda = 1/(1+|\sin\theta_0|)$"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.savefig(FIG_DIR / "grating_lobe_boundary.png")
    plt.close(fig)
    print("[figure] figures/grating_lobe_boundary.png")

    section("PHASE 4 COMPLETE")
    print(f"  results : {out_csv.relative_to(REPO_ROOT)}  ({len(results)} rows)")
    print("  figures : confusion_matrix.png, roc_curves.png, grating_lobe_boundary.png")


if __name__ == "__main__":
    main()
