# 🏦 AI-Powered Loan Risk Underwriting Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-success)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-red)

## 📌 Project Overview
This repository contains an end-to-end Machine Learning pipeline designed to predict loan defaults. Instead of relying on manual bank risk grades (which were intentionally excluded to prevent data leakage), this AI engine evaluates a borrower's raw financial health—such as Debt-to-Income (DTI) ratio and Interest Rate—to make unbiased, high-accuracy underwriting decisions.

The project culminates in a fully interactive **Gradio Web Dashboard** where loan officers can input applicant details and receive instant, probability-scored risk assessments.

## 💼 Business Impact & Value
* **Capital Protection:** By accurately identifying high-risk applicants in the test data, this model successfully shielded an estimated **$3,000,000+** in core capital from being lost to defaults.
* **Risk-Based Pricing:** Proved mathematically that assigning high-interest rates to highly leveraged borrowers (High DTI) triggers a debt spiral.
* **Automated Triage:** Reduces manual underwriting time by instantly approving low-risk applicants and flagging "Danger Zone" profiles for human review.

## 🛠️ Tech Stack
* **Data Processing & ML Pipeline:** `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` (SMOTE)
* **Predictive Algorithm:** `xgboost` (XGBClassifier)
* **Data Visualization:** `matplotlib`, `seaborn`
* **Model Serialization:** `joblib`
* **Interactive UI / MLOps:** `gradio`

## 📂 Repository Structure
```text
├── loan_dataset_20000.csv       # The original historical dataset (20k records)
├── loan_model.ipynb             # Jupyter Notebook with EDA, training, and business insights
├── train.py                     # Python script to instantly train and serialize the XGBoost model
├── app.py                       # Gradio web dashboard application
├── xgboost_loan_model.pkl       # Saved ML pipeline (Generated after running train.py)
└── README.md                    # Project documentation
