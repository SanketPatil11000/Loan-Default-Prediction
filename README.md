# 💳 Loan Default Prediction using Machine Learning

A beginner-friendly Machine Learning college mini-project built with Python, Scikit-learn, and Streamlit. This project predicts whether a loan applicant is likely to default (`1 = Default`, `0 = No Default`), compares three popular classification algorithms, prevents data leakage using Scikit-learn Pipelines, and provides an interactive web dashboard for real-time risk assessment.

---

## 📁 Project Structure

```text
Loan_Default_Prediction/
│
├── data/
│   ├── raw/                        # Untouched, original CSV dataset
│   │   └── README.md               # Dataset download instructions & schema guide
│   └── processed/                  # Cleaned/transformed data splits (if exported)
│       └── .gitkeep
│
├── notebooks/                      # Jupyter notebooks for visual experimentation
│   ├── README.md
│   └── 01_eda_and_exploration.ipynb # (Optional) EDA, distributions & correlation plots
│
├── src/                            # Modular Python source code
│   ├── __init__.py                 # Treats src as a Python package
│   ├── data_loader.py              # Script to load and inspect raw CSV data
│   ├── preprocessing.py            # Train-test split & data leakage-free pipelines
│   ├── train.py                    # Training & comparison of LR, Decision Tree, RF
│   └── evaluate.py                 # Performance metrics (Recall, ROC-AUC, Confusion Matrix)
│
├── models/                         # Serialized best model artifacts (.joblib / .pkl)
│   └── .gitkeep
│
├── app/                            # User Interface
│   └── app.py                      # Interactive Streamlit web application
│
├── requirements.txt                # Required Python libraries and dependencies
├── .gitignore                      # Prevents committing large data files and model binaries
└── README.md                       # Complete project documentation and guide
```

---

## 📄 File & Directory Descriptions

| Directory / File | Description & Purpose |
| :--- | :--- |
| **`data/raw/`** | Stores the original, immutable CSV file (e.g., `loan_data.csv`). Never modify raw data directly. |
| **`data/processed/`** | Holds cached preprocessed datasets or train/test splits if needed for analysis. |
| **`notebooks/`** | Contains Jupyter Notebooks used for Exploratory Data Analysis (EDA), inspecting class distributions, and plotting correlation heatmaps before building production code. |
| **`src/data_loader.py`** | Contains functions (`load_dataset`, `inspect_data`) to read raw CSV files, verify file existence, and output diagnostic summaries. |
| **`src/preprocessing.py`** | Implements train-test splitting and creates Scikit-learn `Pipeline` and `ColumnTransformer` objects (handling imputation, numerical scaling, and categorical encoding) to **prevent data leakage**. |
| **`src/train.py`** | Coordinates the training process. Evaluates candidate models (**Logistic Regression**, **Decision Tree**, and **Random Forest**), selects the best-performing model, and exports it to `models/`. |
| **`src/evaluate.py`** | Computes key binary classification metrics: Accuracy, Precision, Recall (crucial for default detection), F1-Score, ROC-AUC, and Confusion Matrix. |
| **`models/`** | Destination directory where the final trained pipeline artifact (`best_model.joblib`) is saved. |
| **`app/app.py`** | An interactive Streamlit web dashboard. Allows end users or loan officers to modify applicant financial figures (income, credit score, loan amount, DTI) and observe real-time predictions. |
| **`requirements.txt`** | Lists all third-party Python packages and minimum versions required to run the project. |
| **`.gitignore`** | Ensures virtual environments, temporary cache files, model weights, and heavy CSV files are excluded from Git commits. |

---

## 🛡️ Preventing Data Leakage

In financial risk modeling, **data leakage** occurs when information from the test dataset leaks into the training process (e.g., fitting a scaler or imputer on the full dataset before splitting).

To guarantee strict compliance and validity:
1. **Split First**: The dataset is split into training and test sets *before* calculating any statistical properties (mean, median, standard deviation, categories).
2. **Pipelines**: Feature scaling (`StandardScaler`) and missing value imputation (`SimpleImputer`) are bundled inside a Scikit-learn `Pipeline`.
3. **Fit on Train, Transform on Test**: Transformers are only fitted on the training split (`fit_transform`) and subsequently used to transform the test set (`transform`).

---

## 🤖 Models Compared

1. **Logistic Regression**:
   - Baseline linear model.
   - Interpretable coefficients explaining feature influence on log-odds of default.
2. **Decision Tree Classifier**:
   - Captures non-linear relationships and interactions between financial indicators.
   - Simple, human-readable rule splits.
3. **Random Forest Classifier**:
   - Ensemble bagging technique combining multiple decision trees.
   - High predictive accuracy, robust to outliers, and reduces overfitting risk.

---

## 🚀 Setup & Installation Instructions

### 1. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Dataset
Download a Loan Default / Prediction dataset (from Kaggle, e.g., LendingClub or Loan Prediction Problem) and place the CSV in:
```text
data/raw/loan_data.csv
```

### 4. Running the Workflow (Once Dataset is Added)
- **Train and Compare Models**:
  ```bash
  python -m src.train
  ```
- **Launch the Streamlit App**:
  ```bash
  streamlit run app/app.py
  ```
