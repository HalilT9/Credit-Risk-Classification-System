# CreditGuard AI — Credit Risk Classification System

This repository contains a complete Machine Learning project for **credit risk assessment** using the **German Credit dataset (N=1000)**. The system is deployed as a **Streamlit application** and includes training/evaluation code, saved model artifacts, quantitative metrics, visualizations, and the final presentation (PDF).

---

## 1) Description of the Problem

Credit risk assessment is critical for financial institutions because **loan defaults create capital loss**.  
We formulate the task as **binary classification**:

- **Good (Approve)**: low-risk applicant  
- **Bad (Reject)**: high-risk applicant

**Primary goal:** maximize **Recall for the “Bad” class** (risk detection sensitivity).  
Reason: a **false negative** (predicting *Good* for a truly risky borrower) is typically more costly than a false positive in “safety-first” banking policies.

---

## 2) Dataset

- **Dataset:** German Credit Dataset (**N=1000**)
- **Included file:** `german_credit_data.csv`

Target encoding:
- `Risk = 1` → bad (Reject)
- `Risk = 0` → good (Approve)

---

## 3) Preprocessing Procedures

Implemented in `Project.ipynb` and mirrored in the deployed pipeline (`app.py` uses the saved preprocessing artifacts).

- **Missing values handling**
  - `SavingAccounts` and `CheckingAccount` missing values are treated as a separate category: **Unknown/unknown**
- **Encoding**
  - One-Hot Encoding for categorical features
- **Scaling**
  - Numerical features scaled with `StandardScaler` (saved as `scaler.pkl`)
- **Class imbalance**
  - SMOTE is applied on the **training split only** to address minority-class imbalance

Saved artifacts used by the app:
- `scaler.pkl` — fitted scaler
- `model_columns.pkl` — expected one-hot column order
- `final_model.pkl` — trained classifier

---

## 4) Environment / Dependencies

This repository includes an environment specification file:

- `requirements.txt`

Install:
``bash
pip install -r requirements.txt


## 5) Model Details & Methodology
We benchmarked multiple supervised models:

``Logistic Regression``

``Random Forest``

``AdaBoost``

``XGBoost``

``Neural Network (MLP)``

``SVM (Linear) ✅ (final selected model)``



Final model: Support Vector Machine (SVC) with linear kernel.

Motivation: best Recall on the minority class (“Bad”), aligning with the risk-focused objective.

Interpretability:

Feature influence is inspected via SVM coefficients (see shap_grafigi.png).

## 6) Instructions for Execution
A) Run the Streamlit App (Inference)

-``git clone https://github.com/HalilT9/Credit-Risk-Classification-System.git``

``-cd Credit-Risk-Classification-System``

``-pip install -r requirements.txt``

``-streamlit run app.py``



B) Reproduce Experiments (Training + Evaluation)
``Open and run:``
Project.ipynb


This notebook contains:
``
preprocessing steps
model benchmarking
evaluation metrics
generation of plots/images saved in the repository
``

## 7) Model Results —`` Quantitative Evaluation Metrics``

7.1 Final Model (SVM) — ``Test Set Metrics``
```
From the test-set confusion matrix (N=200) (confusion_matrix.png):
Confusion matrix values:

TN=86, FP=54, FN=12, TP=48

Key metrics (Bad = positive class):

Accuracy: 0.67

Bad Recall (Sensitivity): 0.80

Bad Precision: 0.47

Bad F1-score: 0.59

Good Precision: 0.88

Good Recall: 0.61

Good F1-score: 0.72

```

For the full classification report (including macro/weighted averages), see Project.ipynb.

7.2 ROC-AUC Benchmark (Risk Detection Capability)
``
AUC values shown in model_comparison.png:
Logistic Regression: 0.75
Random Forest: 0.76
AdaBoost: 0.75
SVM: 0.77
Neural Network (MLP): 0.68
XGBoost: 0.74
``

## 8) Example Inference Visualizations (Test Set)

The repository includes multiple test/evaluation visualizations:

``confusion_matrix.png`` — confusion matrix on the test set (SVM)
``model_comparison.png`` — model benchmark (Accuracy vs Recall) and ROC curves
``anomaly_graph.png`` — PCA projection (risk as anomaly perspective)
``shap_grafigi.png`` — feature importance / coefficient magnitude (SVM)
``project_pipeline.png ``— system architecture / pipeline overview



## 9) Training Process Plots (Loss / Metric Curves)
SVM is not trained via epoch-based gradient descent, so a classical “loss curve” is not applicable.
However, this repository includes a metric-curve illustrating the improvement of Recall during the tuning/optimization phase:
``genetic_graph.png ``— Recall score trend over optimization iterations (“generations”)
(Details and the code producing this figure are documented in Project.ipynb.)



## 10) Project Presentation (PDF)
credit_risk_classification_presentation.pdf — Final presentation file in PDF format



## 11) Experiment-to-Repository Consistency Statement
All results, metrics, and figures included in the final presentation are produced by the code and experiments documented in this repository, primarily in Project.ipynb, and exported as the image files listed above.



## 12) Repository Contents (Complete Source Code)
``app.py`` — Streamlit inference application

``Project.ipynb ``— complete training + evaluation notebook

``german_credit_data.csv ``— dataset

``requirements.txt`` — environment specification

``final_model.pkl, scaler.pkl, model_columns.pkl, final_model_name.pkl`` — saved artifacts

``*.png ``— evaluation plots and system diagrams

``credit_risk_classification_presentation.pdf ``— final presentation





