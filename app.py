"""
Loan Default Prediction System - Streamlit Web Application
----------------------------------------------------------
College Mini-Project Demonstration
Powered by Scikit-learn Pipeline (Logistic Regression with class_weight='balanced')
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Default Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #92400E;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL ARTIFACT LOADING
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"


@st.cache_resource
def load_trained_pipeline():
    """Loads the serialized Scikit-learn pipeline."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    """Loads saved training metadata and metrics."""
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


pipeline = load_trained_pipeline()
metadata = load_metadata()

# -----------------------------------------------------------------------------
# 3. HEADER & INTRO
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">💳 Loan Default Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Interactive Machine Learning dashboard to evaluate borrower credit risk and predict default probability '
    'using applicant financial and demographic characteristics.'
    '</div>',
    unsafe_allow_html=True
)

if pipeline is None:
    st.error("⚠️ Model pipeline artifact not found at `models/best_model.joblib`.")
    st.info("Please train the model first by running `python -m src.train` in your terminal.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. SIDEBAR - PROFILE PRESETS & APP INFO
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Profile Presets")
preset = st.sidebar.selectbox(
    "Load Sample Applicant Profile:",
    ("Custom Profile", "Prime Borrower (Low Risk)", "Subprime Borrower (High Risk)")
)

# Preset default values
if preset == "Prime Borrower (Low Risk)":
    default_age = 52
    default_income = 115000
    default_loan = 35000
    default_credit = 760
    default_months_emp = 84
    default_credit_lines = 2
    default_interest = 5.25
    default_term = 36
    default_dti = 0.22
    default_edu = "Master's"
    default_emp = "Full-time"
    default_marital = "Married"
    default_mortgage = "No"
    default_dependents = "Yes"
    default_purpose = "Home"
    default_cosigner = "Yes"
elif preset == "Subprime Borrower (High Risk)":
    default_age = 23
    default_income = 24000
    default_loan = 145000
    default_credit = 440
    default_months_emp = 6
    default_credit_lines = 4
    default_interest = 21.50
    default_term = 60
    default_dti = 0.78
    default_edu = "High School"
    default_emp = "Unemployed"
    default_marital = "Single"
    default_mortgage = "Yes"
    default_dependents = "No"
    default_purpose = "Other"
    default_cosigner = "No"
else:
    default_age = 38
    default_income = 68000
    default_loan = 45000
    default_credit = 650
    default_months_emp = 36
    default_credit_lines = 2
    default_interest = 11.50
    default_term = 36
    default_dti = 0.35
    default_edu = "Bachelor's"
    default_emp = "Full-time"
    default_marital = "Married"
    default_mortgage = "No"
    default_dependents = "No"
    default_purpose = "Auto"
    default_cosigner = "No"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Active Model Pipeline")
if metadata:
    st.sidebar.write(f"**Algorithm:** `{metadata.get('selected_model', 'Logistic Regression')}`")
    metrics = metadata.get("metrics", {})
    st.sidebar.write(f"• **Test Recall:** `{metrics.get('recall', 0):.2%}`")
    st.sidebar.write(f"• **ROC-AUC:** `{metrics.get('roc_auc', 0):.3f}`")
    st.sidebar.write(f"• **Accuracy:** `{metrics.get('accuracy', 0):.2%}`")

# -----------------------------------------------------------------------------
# 5. INPUT FORM TABS / COLUMNS
# -----------------------------------------------------------------------------
st.subheader("📋 Applicant Financial & Demographic Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Demographics & Education")
    age = st.slider("Age", min_value=18, max_value=80, value=int(default_age), step=1)
    education = st.selectbox(
        "Education Level",
        ["Bachelor's", "High School", "Master's", "PhD"],
        index=["Bachelor's", "High School", "Master's", "PhD"].index(default_edu)
    )
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"],
        index=["Single", "Married", "Divorced"].index(default_marital)
    )
    has_dependents = st.selectbox(
        "Has Dependents?",
        ["No", "Yes"],
        index=["No", "Yes"].index(default_dependents)
    )

with col2:
    st.markdown("#### 💼 Employment & Income")
    employment_type = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Self-employed", "Unemployed"],
        index=["Full-time", "Part-time", "Self-employed", "Unemployed"].index(default_emp)
    )
    months_employed = st.number_input(
        "Months Employed",
        min_value=0, max_value=180, value=int(default_months_emp), step=1
    )
    income = st.number_input(
        "Annual Income ($)",
        min_value=5000, max_value=1000000, value=int(default_income), step=1000
    )
    dti_ratio = st.slider(
        "Debt-to-Income (DTI) Ratio",
        min_value=0.05, max_value=1.00, value=float(default_dti), step=0.01,
        help="Monthly debt payments divided by gross monthly income."
    )

with col3:
    st.markdown("#### 💰 Loan & Credit Terms")
    credit_score = st.slider(
        "Credit Score",
        min_value=300, max_value=850, value=int(default_credit), step=1,
        help="Credit score rating (300-850)."
    )
    loan_amount = st.number_input(
        "Requested Loan Amount ($)",
        min_value=1000, max_value=500000, value=int(default_loan), step=1000
    )
    interest_rate = st.slider(
        "Interest Rate (%)",
        min_value=1.0, max_value=30.0, value=float(default_interest), step=0.25
    )
    loan_term = st.selectbox(
        "Loan Term (Months)",
        [12, 24, 36, 48, 60],
        index=[12, 24, 36, 48, 60].index(default_term)
    )
    num_credit_lines = st.slider(
        "Number of Credit Lines",
        min_value=1, max_value=10, value=int(default_credit_lines), step=1
    )
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Auto", "Business", "Education", "Home", "Other"],
        index=["Auto", "Business", "Education", "Home", "Other"].index(default_purpose)
    )
    has_mortgage = st.selectbox(
        "Has Existing Mortgage?",
        ["No", "Yes"],
        index=["No", "Yes"].index(default_mortgage)
    )
    has_cosigner = st.selectbox(
        "Has Co-Signer?",
        ["No", "Yes"],
        index=["No", "Yes"].index(default_cosigner)
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. PREDICTION LOGIC & PRESENTATION
# -----------------------------------------------------------------------------
if st.button("🔍 Assess Default Risk", type="primary", use_container_width=True):
    # Construct DataFrame with column names and types matching training set exactly
    applicant_data = pd.DataFrame([{
        "Age": int(age),
        "Income": int(income),
        "LoanAmount": int(loan_amount),
        "CreditScore": int(credit_score),
        "MonthsEmployed": int(months_employed),
        "NumCreditLines": int(num_credit_lines),
        "InterestRate": float(interest_rate),
        "LoanTerm": int(loan_term),
        "DTIRatio": float(dti_ratio),
        "Education": str(education),
        "EmploymentType": str(employment_type),
        "MaritalStatus": str(marital_status),
        "HasMortgage": str(has_mortgage),
        "HasDependents": str(has_dependents),
        "LoanPurpose": str(loan_purpose),
        "HasCoSigner": str(has_cosigner)
    }])

    # Run inference through complete preprocessor + model pipeline
    prediction = pipeline.predict(applicant_data)[0]
    probabilities = pipeline.predict_proba(applicant_data)[0] if hasattr(pipeline, "predict_proba") else [0.5, 0.5]
    default_prob = probabilities[1]
    repay_prob = probabilities[0]

    st.markdown("### 🎯 Assessment Results")

    res_col1, res_col2 = st.columns([1.2, 1.8])

    with res_col1:
        if prediction == 1:
            st.error("### 🚨 High Risk: Default Likely (Class 1)")
            st.write(f"The model flags this applicant as likely to **default** on the loan.")
        else:
            st.success("### ✅ Low Risk: Repayment Likely (Class 0)")
            st.write(f"The model evaluates this applicant as **creditworthy** (No Default).")

        st.metric(
            label="Estimated Default Probability",
            value=f"{default_prob * 100:.2f}%",
            delta=f"{(default_prob - 0.5) * 100:+.1f}% vs. balanced threshold",
            delta_color="inverse" if prediction == 1 else "normal"
        )
        st.progress(float(default_prob))

    with res_col2:
        st.markdown("#### 🔎 Result Explanation & Risk Factor Insights")
        
        reasons = []
        if dti_ratio > 0.45:
            reasons.append(f"⚠️ **High Debt-to-Income ({dti_ratio:.0%})**: Heavy existing debt burden reduces monthly repayment capacity.")
        else:
            reasons.append(f"✔️ **Healthy DTI ({dti_ratio:.0%})**: Debt burden is within safe operational thresholds.")

        if credit_score < 580:
            reasons.append(f"⚠️ **Subprime Credit Score ({credit_score})**: Indicates higher historical credit delinquency.")
        elif credit_score >= 700:
            reasons.append(f"✔️ **Prime Credit Score ({credit_score})**: Strong credit history supports creditworthiness.")

        if interest_rate >= 16.0:
            reasons.append(f"⚠️ **High Interest Rate ({interest_rate:.2f}%)**: Elevated interest rate compounds payment risk.")

        if income > 80000:
            reasons.append(f"✔️ **Substantial Annual Income (${income:,})**: Strong income buffer protects against financial shocks.")

        if has_cosigner == "Yes":
            reasons.append("✔️ **Co-Signer Present**: Reduces lender exposure through secondary liability.")

        if employment_type == "Unemployed":
            reasons.append("⚠️ **Unemployed Status**: Lack of recurring earned income presents substantial default risk.")

        for r in reasons:
            st.write(r)

# -----------------------------------------------------------------------------
# 7. EDUCATIONAL DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Educational Project Disclaimer:</strong><br>
    This web application is an educational Machine Learning college mini-project demonstration.
    The predictions and probabilities generated are based strictly on statistical patterns in synthetic/sample data
    and <strong>do not constitute actual banking, lending, or legal credit underwriting decisions</strong>.
</div>
""", unsafe_allow_html=True)
