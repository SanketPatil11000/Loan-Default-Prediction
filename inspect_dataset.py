"""
Inspect Dataset Script
----------------------
Loads and inspects the Loan Default dataset without modifying the source CSV or training models.

Checks performed:
1. Data loading and row/column shape.
2. First 5 rows display.
3. Complete column listing and data types.
4. Missing value checks per column.
5. Duplicate record detection.
6. Target distribution ('Default' column value counts and proportions).
7. Diagnostic summary and key takeaways.
"""

from pathlib import Path
import pandas as pd


def inspect_loan_dataset():
    # 1. Resolve dataset file path robustly
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / "dataset" / "Loan_default.csv"

    print("=" * 80)
    print("📊 LOAN DEFAULT DATASET INSPECTION")
    print("=" * 80)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at expected path: {file_path}")

    print(f"\n[1] Loading dataset from: {file_path}")
    # Load dataset (read-only)
    df = pd.read_csv(file_path)
    print("    --> Successfully loaded into memory.")

    # 2. Number of rows and columns
    num_rows, num_cols = df.shape
    print(f"\n[2] Dataset Shape:")
    print(f"    - Total Rows (Records):    {num_rows:,}")
    print(f"    - Total Columns (Features): {num_cols}")

    # 3. Display first 5 rows
    print(f"\n[3] First 5 Rows Preview:")
    print("-" * 80)
    # Set display options so all columns are visible in console
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head())
    print("-" * 80)

    # 4. Display all column names and data types
    print(f"\n[4] Columns and Data Types:")
    print("-" * 80)
    col_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.values,
        "Non-Null Count": df.notnull().sum().values
    })
    print(col_info.to_string(index=False))

    # 5. Check missing values
    print(f"\n[5] Missing Values Check:")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if missing_cols.empty:
        print("    ✅ No missing (null) values found across any columns!")
    else:
        print("    ⚠️ Found missing values in the following columns:")
        for col, count in missing_cols.items():
            pct = (count / num_rows) * 100
            print(f"       - {col}: {count:,} ({pct:.2f}%)")

    # 6. Check duplicate records
    print(f"\n[6] Duplicate Records Check:")
    print("-" * 80)
    num_duplicates = df.duplicated().sum()
    if num_duplicates == 0:
        print("    ✅ No duplicate rows found in the dataset.")
    else:
        pct_dup = (num_duplicates / num_rows) * 100
        print(f"    ⚠️ Duplicate rows found: {num_duplicates:,} ({pct_dup:.2f}%)")

    # 7. Value counts of Default target column
    print(f"\n[7] Target Column ('Default') Distribution:")
    print("-" * 80)
    
    # Locate target column (handling potential case variations)
    target_col = None
    for col in df.columns:
        if col.strip().lower() == "default":
            target_col = col
            break

    if target_col is None:
        print("    ❌ Warning: 'Default' column was not found. Available columns:")
        print(f"       {list(df.columns)}")
    else:
        counts = df[target_col].value_counts().sort_index()
        proportions = df[target_col].value_counts(normalize=True).sort_index() * 100
        summary_target = pd.DataFrame({
            "Label Meaning": ["0 = No Default", "1 = Default"],
            "Count": counts.values,
            "Percentage (%)": proportions.values
        })
        print(summary_target.to_string(index=False))

        # Check for class imbalance
        imbalance_ratio = counts.iloc[0] / counts.iloc[1] if len(counts) > 1 and counts.iloc[1] > 0 else 0
        print(f"\n    Class Imbalance Ratio (No Default : Default) = ~{imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2.0:
            print("    💡 Note: The dataset exhibits class imbalance (fewer defaults than non-defaults).")
            print("       We should use stratified splitting and appropriate metrics (Recall, ROC-AUC) or class weighting.")

    # 8. Clear summary of dataset
    print(f"\n[8] Summary of the Dataset:")
    print("=" * 80)
    print(f"  • Size: {num_rows:,} applicant records across {num_cols} features.")
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string', 'str']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    print(f"  • Categorical / Text columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"  • Numerical columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"  • Data cleanliness: {'Clean (no missing or duplicate records)' if missing_cols.empty and num_duplicates == 0 else 'Requires cleaning'}.")
    print("=" * 80)


if __name__ == "__main__":
    inspect_loan_dataset()
