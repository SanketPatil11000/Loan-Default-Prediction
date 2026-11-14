"""
Machine Learning Training & Model Selection Pipeline
-----------------------------------------------------
Trains and evaluates 4 models:
1. Dummy Classifier (Baseline)
2. Logistic Regression (Linear with class_weight='balanced')
3. Decision Tree Classifier (Tree-based with class_weight='balanced')
4. Random Forest Classifier (Ensemble with class_weight='balanced')

All models are paired with the ColumnTransformer inside a Scikit-learn Pipeline
to guarantee zero data leakage. Evaluates all models, compares results, selects
the top-performing model, saves the pipeline to models/best_model.joblib, and verifies
end-to-end inference.
"""

from pathlib import Path
import json
import time
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Internal modular imports
from src.data_loader import load_dataset
from src.preprocessing import prepare_features_and_split, build_preprocessor
from src.evaluate import (
    evaluate_model_performance,
    generate_comparison_table,
    plot_and_save_visualizations
)


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiates candidate classification models.
    class_weight='balanced' is applied to penalize false negatives on the minority default class.
    """
    return {
        "Dummy Classifier (Baseline)": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=15,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        )
    }


def select_best_model(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Selects the best model based on default-class Recall, F1-Score, and ROC-AUC.
    Ignores Dummy Classifier.
    """
    # Filter out baseline Dummy Classifier
    candidates = [r for r in results_list if "Dummy" not in r["Model"]]

    # Score: Weighted combination emphasizing Recall (40%), ROC-AUC (35%), and F1 (25%)
    best_candidate = max(
        candidates,
        key=lambda r: (
            0.40 * r["Recall"] +
            0.35 * (r["ROC-AUC"] if r["ROC-AUC"] is not None else 0.5) +
            0.25 * r["F1-Score"]
        )
    )
    return best_candidate


def verify_saved_pipeline(model_path: Path, X_sample: pd.DataFrame) -> None:
    """
    Verifies that the serialized pipeline loads properly from disk and can run inferences.
    """
    print("\n" + "=" * 80)
    print("🔍 VERIFYING SAVED PIPELINE INFERENCE")
    print("=" * 80)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing at: {model_path}")

    loaded_pipeline = joblib.load(model_path)
    print(f"[INFO] Pipeline loaded successfully from: {model_path.name}")

    # Predict on a sample of test records
    predictions = loaded_pipeline.predict(X_sample)
    probabilities = loaded_pipeline.predict_proba(X_sample)[:, 1] if hasattr(loaded_pipeline, "predict_proba") else None

    print(f"\n[INFO] Sample Test Inferences (first {len(X_sample)} records):")
    sample_results = pd.DataFrame({
        "Sample Index": X_sample.index,
        "Predicted Class": predictions,
        "Default Risk Probability (%)": [f"{p*100:.2f}%" for p in probabilities] if probabilities is not None else ["N/A"] * len(X_sample)
    })
    print(sample_results.to_string(index=False))
    print("\n✅ Verification Successful: Saved pipeline accepts raw data and outputs valid predictions!")


def run_training_pipeline():
    start_time = time.time()
    base_dir = Path(__file__).resolve().parent.parent
    outputs_dir = base_dir / "outputs"
    graphs_dir = outputs_dir / "model_evaluation"
    models_dir = base_dir / "models"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 LOAN DEFAULT PREDICTION - MACHINE LEARNING PIPELINE")
    print("=" * 80)

    # 1. Load data
    df = load_dataset()

    # 2. Split data and isolate features (stratified, random_state=42, dropping LoanID)
    X_train, X_test, y_train, y_test, num_features, cat_features = prepare_features_and_split(
        df,
        target_column="Default",
        id_column="LoanID",
        test_size=0.2,
        random_state=42
    )

    # 3. Build ColumnTransformer preprocessor
    preprocessor = build_preprocessor(num_features, cat_features)

    # 4. Get candidate models
    models_dict = get_candidate_models(random_state=42)

    # 5. Train and evaluate each model
    results_list = []
    trained_pipelines = {}

    print("\n" + "=" * 80)
    print("🏋️ TRAINING AND EVALUATING CANDIDATE MODELS")
    print("=" * 80)

    for model_name, classifier in models_dict.items():
        print(f"\n--> Training: {model_name}...")
        t0 = time.time()

        # Connect preprocessor + classifier into a unified Pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        # Fit strictly on training data (no data leakage)
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"    Completed in {elapsed:.2f} seconds.")

        # Evaluate performance on test data
        eval_result = evaluate_model_performance(model_name, pipeline, X_test, y_test)
        results_list.append(eval_result)
        trained_pipelines[model_name] = pipeline

        print(f"    [Metrics] Accuracy: {eval_result['Accuracy']:.4f} | "
              f"Recall (Class 1): {eval_result['Recall']:.4f} | "
              f"Precision: {eval_result['Precision']:.4f} | "
              f"F1: {eval_result['F1-Score']:.4f} | "
              f"ROC-AUC: {eval_result['ROC-AUC']:.4f}")

    # 6. Generate comparison table
    df_comparison = generate_comparison_table(results_list)
    print("\n" + "=" * 80)
    print("📊 MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    print(df_comparison.to_string(index=False))

    # Save comparison table to CSV and Text
    csv_path = outputs_dir / "model_comparison.csv"
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved comparison CSV to: {csv_path.name}")

    # 7. Generate and save evaluation plots
    print("\n[INFO] Generating evaluation charts in outputs/model_evaluation/...")
    plot_and_save_visualizations(results_list, y_test, graphs_dir)

    # 8. Select best model based on Recall, F1, and ROC-AUC
    best_result = select_best_model(results_list)
    best_model_name = best_result["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    print("\n" + "=" * 80)
    print(f"🏆 BEST MODEL SELECTED: {best_model_name}")
    print("=" * 80)
    print(f"  • Recall (Default Class 1):  {best_result['Recall']:.4f}")
    print(f"  • ROC-AUC Score:             {best_result['ROC-AUC']:.4f}")
    print(f"  • F1-Score:                  {best_result['F1-Score']:.4f}")
    print(f"  • Overall Accuracy:          {best_result['Accuracy']:.4f}")

    # 9. Save the complete best pipeline
    model_save_path = models_dir / "best_model.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"\n[INFO] Complete best pipeline exported to: {model_save_path}")

    # 10. Save metadata and summary
    metadata = {
        "selected_model": best_model_name,
        "metrics": {
            "accuracy": best_result["Accuracy"],
            "precision": best_result["Precision"],
            "recall": best_result["Recall"],
            "f1_score": best_result["F1-Score"],
            "roc_auc": best_result["ROC-AUC"]
        },
        "features": {
            "numerical": num_features,
            "categorical": cat_features
        },
        "target": "Default",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[INFO] Saved model metadata to: {metadata_path.name}")

    summary_path = outputs_dir / "best_model_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"BEST MODEL SUMMARY\n")
        f.write(f"==================\n")
        f.write(f"Selected Model: {best_model_name}\n")
        f.write(f"Accuracy:       {best_result['Accuracy']:.4f}\n")
        f.write(f"Precision:      {best_result['Precision']:.4f}\n")
        f.write(f"Recall:         {best_result['Recall']:.4f}\n")
        f.write(f"F1-Score:       {best_result['F1-Score']:.4f}\n")
        f.write(f"ROC-AUC:        {best_result['ROC-AUC']:.4f}\n\n")
        f.write("Full Comparison:\n")
        f.write(df_comparison.to_string(index=False))

    # 11. Verify saved pipeline on test sample
    verify_saved_pipeline(model_save_path, X_test.iloc[:5])

    total_time = time.time() - start_time
    print(f"\n🎉 Entire training pipeline completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    run_training_pipeline()
