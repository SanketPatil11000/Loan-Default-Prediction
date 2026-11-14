"""
Model Evaluation Module
-----------------------
Provides metrics computation, comparison table generation, and visualization
helpers for evaluating credit risk classification models.

Metrics tracked:
- Accuracy
- Precision (Default class 1)
- Recall (Default class 1)
- F1-Score (Default class 1)
- ROC-AUC
- Confusion Matrix
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)


def evaluate_model_performance(
    model_name: str,
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates a trained pipeline on test data and returns a dictionary of metrics.
    """
    y_pred = pipeline.predict(X_test)

    # Get predicted probabilities if supported
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_proba = pipeline.decision_function(X_test)
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            roc_auc = 0.5
    else:
        roc_auc = 0.5

    cm = confusion_matrix(y_test, y_pred)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Confusion_Matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def generate_comparison_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Builds a summary comparison DataFrame from a list of model evaluation results.
    """
    summary_rows = []
    for r in results_list:
        summary_rows.append({
            "Model": r["Model"],
            "Accuracy": round(r["Accuracy"], 4),
            "Precision (Class 1)": round(r["Precision"], 4),
            "Recall (Class 1)": round(r["Recall"], 4),
            "F1-Score (Class 1)": round(r["F1-Score"], 4),
            "ROC-AUC": round(r["ROC-AUC"], 4) if r["ROC-AUC"] is not None else 0.5
        })
    df_compare = pd.DataFrame(summary_rows)
    return df_compare


def plot_and_save_visualizations(
    results_list: List[Dict[str, Any]],
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """
    Generates and saves:
    1. Metric comparison bar charts.
    2. Combined ROC Curves.
    3. Confusion Matrix heatmaps for all candidate models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Metrics Comparison Bar Chart
    df_compare = generate_comparison_table(results_list)
    df_melted = df_compare.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision (Class 1)", "Recall (Class 1)", "F1-Score (Class 1)", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="tab10", ax=ax)
    ax.set_title("Candidate Models - Performance Comparison Across Key Metrics", fontsize=14, weight="bold", pad=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_xlabel("Evaluation Metric", fontsize=12)
    ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    chart_path = output_dir / "01_model_comparison_metrics.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved metrics comparison chart -> {chart_path.name}")

    # 2. Combined ROC Curves Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results_list:
        if r["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
            auc_val = r["ROC-AUC"]
            ax.plot(fpr, tpr, label=f"{r['Model']} (AUC = {auc_val:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", linewidth=1.5)
    ax.set_title("ROC Curves Comparison", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = output_dir / "02_roc_curves_comparison.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved ROC curves comparison -> {roc_path.name}")

    # 3. Confusion Matrix Grid Plot
    num_models = len(results_list)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4.5))
    if num_models == 1:
        axes = [axes]

    for idx, r in enumerate(results_list):
        cm = r["Confusion_Matrix"]
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"]
        )
        ax.set_title(f"{r['Model']}\nRecall: {r['Recall']:.1%}", fontsize=12, weight="bold")
        ax.set_ylabel("Actual Label" if idx == 0 else "")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    cm_path = output_dir / "03_confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrices grid -> {cm_path.name}")
