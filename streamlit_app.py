import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

st.markdown('<p style="font-size:30px; font-weight:bold; color:#2E86C1;">📊 Customer Churn Prediction</p>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)

    with col2:
        st.subheader("Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.divider()
    st.subheader("Billing & Contract")
    c3, c4, c5 = st.columns(3)
    contract = c3.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = c4.selectbox("Paperless Billing", ["Yes", "No"])
    method = c5.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    submit = st.form_submit_button("Predict Churn")

if submit:
    # Construct payload to match FastAPI CustomerData class
    payload = {
        "gender": gender, "SeniorCitizen": int(senior), "Partner": partner,
        "Dependents": dependents, "tenure": int(tenure), "PhoneService": phone,
        "MultipleLines": multiple, "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": method,
        "MonthlyCharges": float(monthly), "TotalCharges": float(total)
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            headers={"x-api-key": "ankush_super_secure_key_123"} # Match your .env API_KEY
        )

        if response.status_code == 200:
            res = response.json()
            if res["prediction"] == 1:
                st.error(f"⚠️ Likely to Churn (Prob: {res['probability']:.4f})")
            else:
                st.success(f"✅ Likely to Stay (Prob: {res['probability']:.4f})")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Connection Failed: {e}")