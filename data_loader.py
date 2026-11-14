"""
Data Loader Module
------------------
Responsible for loading the loan default dataset safely from dataset/ or data/raw/.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the loan default dataset.
    Checks dataset/Loan_default.csv first, then data/raw/.

    Parameters:
        file_path (str, optional): Custom path to CSV.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if file_path is not None:
        target_path = Path(file_path)
    else:
        base_dir = Path(__file__).resolve().parent.parent
        possible_paths = [
            base_dir / "dataset" / "Loan_default.csv",
            base_dir / "dataset" / "loan_default.csv",
            base_dir / "data" / "raw" / "Loan_default.csv",
            base_dir / "data" / "raw" / "loan_data.csv"
        ]
        target_path = None
        for p in possible_paths:
            if p.exists():
                target_path = p
                break

        if target_path is None:
            raise FileNotFoundError(
                f"Dataset not found! Checked locations:\n" +
                "\n".join([f" - {p}" for p in possible_paths])
            )

    print(f"[INFO] Loading dataset from: {target_path}")
    df = pd.read_csv(target_path)
    print(f"[INFO] Successfully loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df
