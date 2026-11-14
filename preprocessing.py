"""
Preprocessing Module
--------------------
Handles:
1. Feature extraction and exclusion of identifiers (LoanID).
2. Train-Test Splitting (80% train, 20% test, stratified, random_state=42).
3. Leakage-free preprocessing pipelines (imputation, scaling, one-hot encoding).

Data Leakage Prevention:
------------------------
Transformers (StandardScaler, OneHotEncoder, SimpleImputer) are encapsulated inside
a Scikit-learn Pipeline / ColumnTransformer. They are strictly fitted ONLY on the
training fold (`fit`) and then transform test samples (`transform`).
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def prepare_features_and_split(
    df: pd.DataFrame,
    target_column: str = "Default",
    id_column: str = "LoanID",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Drops the identifier column, isolates features and target, performs stratified
    train-test split, and identifies numerical vs categorical feature names.

    Returns:
        X_train, X_test, y_train, y_test, numeric_features, categorical_features
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe columns: {list(df.columns)}")

    # Drop identifier column if present
    drop_cols = [target_column]
    if id_column in df.columns:
        drop_cols.append(id_column)
        print(f"[INFO] Removed '{id_column}' from modeling features.")

    X = df.drop(columns=drop_cols)
    y = df[target_column]

    # Perform stratified train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[INFO] Stratified Split: Train={len(X_train):,} samples (80%), Test={len(X_test):,} samples (20%)")
    print(f"       Train target balance: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"       Test target balance:  {y_test.value_counts(normalize=True).to_dict()}")

    # Dynamically detect numerical and categorical columns
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "string", "category", "str"]).columns.tolist()

    print(f"[INFO] Identified {len(numeric_features)} numerical features: {numeric_features}")
    print(f"[INFO] Identified {len(categorical_features)} categorical features: {categorical_features}")

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer:
    - Numerical: SimpleImputer(median) -> StandardScaler()
    - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor
