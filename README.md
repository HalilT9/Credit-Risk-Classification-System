#  CreditGuard AI: Credit Risk Classification System

## 1. Description of the Problem
Credit risk assessment is a critical challenge for financial institutions. The primary goal of this project is to minimize **Capital Loss** (financial loss due to loan defaults).

Our objective is to build a Machine Learning model that identifies "Bad" (Risk) customers with high sensitivity (**Recall**), ensuring that potential defaulters are rejected to protect the bank's assets.

## 2. Dataset and Preprocessing
**Dataset:** German Credit Dataset (N=1000).
* **Imbalance Handling:** Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.
* **Missing Values:** 'Saving accounts' and 'Checking account' missing values were treated as a separate category ("unknown"), which proved to be a significant predictor.
* **Encoding:** One-Hot Encoding applied to categorical variables.
* **Scaling:** Numerical features standardized using `StandardScaler`.

## 3. Methodology & Model Architecture
We benchmarked multiple algorithms including **Random Forest, XGBoost, and SVM**.

* **Selected Model:** SVM (Support Vector Classifier) with Linear Kernel.
* **Motivation:** SVM provided the best **Recall (approx. 80%)** for the minority class, aligning with our "Safety First" banking strategy.
* **Dominant Feature:** Checking Account Status was found to be the strongest predictor of risk.

## 4. How to Run (Local Execution)

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPO_LINK_HERE]
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 5. Files Description
* `app.py`: Streamlit web application.
* `Project.ipynb`: Training and Evaluation Notebook.
* `Credit_Risk_Presentation.pdf`: Project presentation slides.
* `*.pkl`: Serialized model files.

## 6. Evaluation Metrics & Visualizations (Requirement Mapping)
In accordance with the project requirements, the following visualizations are provided in the `Credit_Risk_Presentation.pdf` and `Project.ipynb`:

* **Quantitative Metrics:**
    * Full `Classification Report` (Precision, Recall, F1-Score) is available in the Notebook.
    * **Primary Metric:** Recall (0.80) to minimize financial risk.

* **Inference Visualizations:**
    * **Confusion Matrix:** Visualizes the TP/TN/FP/FN distribution on the Test Set.
    * **PCA Projection:** Visualizes the test set distribution, identifying risky customers as statistical anomalies.

* **Training Process Plot:**
    * Since SVM is not an iterative Deep Learning model, standard "Loss Curves" do not apply.
    * **Evolutionary Optimization Curve**: We provide a plot demonstrating the improvement of the **Recall Score over generations** during the Hyperparameter Tuning process (See Presentation Slide: "Evolutionary Optimization").