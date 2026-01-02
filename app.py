
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="CreditGuard AI Risk System", page_icon="🏦", layout="wide")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('final_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except FileNotFoundError:
        return None, None, None

model, scaler, model_columns = load_assets()

st.title("🏦 CreditGuard AI: Real-Time Credit Risk Analysis")
st.markdown("**Model:** SVM (Linear Kernel) | **Strategy:** Safety First (High Recall)")
st.markdown("---")

st.sidebar.header("Customer Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
credit_amount = st.sidebar.number_input("Credit Amount (DM)", min_value=0, max_value=20000, value=5000)
duration = st.sidebar.slider("Duration (Months)", min_value=1, max_value=72, value=24)

sex = st.sidebar.selectbox("Sex", ["male", "female"])
housing = st.sidebar.selectbox("Housing", ["own", "rent", "free"])
saving = st.sidebar.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich", "Unknown"])
checking = st.sidebar.selectbox("Checking Account", ["little", "moderate", "rich", "Unknown"])
purpose = st.sidebar.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"])

if st.button("ANALYZE RISK", type="primary"):
    if model is not None:
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'Job': [2],
            'Housing': [housing],
            'SavingAccounts': [saving],
            'CheckingAccount': [checking],
            'CreditAmount': [credit_amount],
            'Duration': [duration],
            'Purpose': [purpose]
        })

        st.write("### Input Data")
        st.dataframe(input_data)

        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        input_scaled = scaler.transform(input_encoded)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        risk_prob = probability[0][1]

        st.markdown("### 📊 Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction[0] == 1:
                st.error("⚠️ RESULT: HIGH RISK (REJECT)")
                st.write(f"**Risk Score:** {risk_prob*100:.2f}%")
                st.write("Recommendation: **Do Not Approve**")
            else:
                st.success("✅ RESULT: LOW RISK (APPROVE)")
                st.write(f"**Confidence Score:** {(1-risk_prob)*100:.2f}%")
                st.write("Recommendation: **Approve Loan**")

        with col2:
            st.write("Risk Probability:")
            st.progress(int(risk_prob * 100))
    else:
        st.error("Model files not found. Please run the training notebook first.")
