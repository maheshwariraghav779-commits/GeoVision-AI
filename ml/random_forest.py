# ============================================================
# GEOVISION AI
# RANDOM FOREST - ML EXPERIMENT & MODEL TRAINING
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def find_dataset():
    """Locate the dataset file across common project locations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "geovision_ml_dataset_2023_2026.csv"),
        os.path.join(script_dir, "geovision_ml_dataset_2023_2026 (1).csv"),
        os.path.join(script_dir, "..", "geovision_ml_dataset_2023_2026.csv"),
        os.path.join(script_dir, "..", "geovision_ml_dataset_2023_2026 (1).csv"),
        os.path.join(os.getcwd(), "geovision_ml_dataset_2023_2026.csv"),
        os.path.join(os.getcwd(), "geovision_ml_dataset_2023_2026 (1).csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Could not find the dataset. Searched locations:\n" +
        "\n".join(f" - {p}" for p in candidates)
    )

def main():
    print("======================================")
    print("GEOVISION AI - RANDOM FOREST")
    print("======================================")

    # ============================================================
    # 1. LOAD DATASET
    # ============================================================
    dataset_path = find_dataset()
    print(f"\n[1] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())

    # ============================================================
    # 2. CHECK DATA INTEGRITY & LABEL DISTRIBUTION
    # ============================================================
    null_counts = df.isnull().sum().sum()
    print(f"\n[2] Data Integrity Check:")
    print(f"Total missing (NaN) values in dataset: {null_counts}")

    print("\nLabel distribution:")
    print(df["Label"].value_counts())

    # ============================================================
    # 3. REMOVE NON-ML COLUMNS
    # ============================================================
    # system:index = GEE sample identifier
    # .geo = geometry information
    columns_to_remove = ["system:index", ".geo"]
    df = df.drop(columns=columns_to_remove, errors="ignore")

    # ============================================================
    # 4. SEPARATE FEATURES AND TARGET
    # ============================================================
    X = df.drop(columns=["Label"])
    y = df["Label"]

    # ============================================================
    # 5. CHECK FEATURES
    # ============================================================
    print("\n[5] Features used by the model:")
    for feature in X.columns:
        print(f" - {feature}")
    print(f"Total number of features: {X.shape[1]}")

    # ============================================================
    # 6. TRAIN / TEST SPLIT
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\n[6] Split Summary:")
    print(f"Training samples: {len(X_train)} (80%)")
    print(f"Testing samples : {len(X_test)} (20%)")

    # ============================================================
    # 7. CREATE RANDOM FOREST MODEL
    # ============================================================
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # ============================================================
    # 8. 5-FOLD CROSS-VALIDATION
    # ============================================================
    print("\n[7 & 8] Running 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Per-fold scores: {[round(s, 4) for s in cv_scores]}")

    # ============================================================
    # 9. TRAIN FINAL MODEL ON TRAIN SET
    # ============================================================
    print("\nTraining Random Forest model on train split...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # ============================================================
    # 10. PREDICTIONS & PROBABILITIES
    # ============================================================
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ============================================================
    # 11. MODEL EVALUATION
    # ============================================================
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n======================================")
    print("MODEL PERFORMANCE (TEST SET)")
    print("======================================")
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"ROC-AUC Score  : {roc_auc:.4f}")

    # ============================================================
    # 12. CLASSIFICATION REPORT & CONFUSION MATRIX
    # ============================================================
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ============================================================
    # 13. FEATURE IMPORTANCE
    # ============================================================
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\n======================================")
    print("FEATURE IMPORTANCE")
    print("======================================")
    print(importance.to_string(index=False))

    # ============================================================
    # 14. SAVE ARTIFACTS IN ML FOLDER
    # ============================================================
    # Determine ml output directory (avoid ml/ml if script is already in ml/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir if os.path.basename(script_dir).lower() == "ml" else os.path.join(script_dir, "ml")
    os.makedirs(output_dir, exist_ok=True)

    importance_path = os.path.join(output_dir, "feature_importance.csv")
    importance.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to: {importance_path}")

    model_path = os.path.join(output_dir, "geovision_rf_model.joblib")
    joblib.dump(model, model_path)
    print(f"Trained model saved to: {model_path}")

    # Try saving visual plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Plot Feature Importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance["Feature"][::-1], importance["Importance"][::-1], color="#2563eb")
        plt.xlabel("Importance")
        plt.title("GeoVision Random Forest - Feature Importance")
        plt.tight_layout()
        chart_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Feature importance chart saved to: {chart_path}")
    except Exception as e:
        print(f"Note: Visual chart not generated ({e})")

    print("\n======================================")
    print("GEOVISION AI ML EXPERIMENT COMPLETE")
    print("======================================")

if __name__ == "__main__":
    main()
    