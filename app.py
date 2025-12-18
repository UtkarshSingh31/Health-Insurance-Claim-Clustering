import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

output_dir = "E:\\Fraud Detection\\models"

st.title("Healthcare Fraud Detection System")
st.write("Enter claim details to check if it's suspicious")

# Input form
claim_amount = st.number_input("Claim Amount ($)", min_value=100, max_value=100000, value=5000)
claim_type = st.selectbox("Claim Type", ["emergency", "hospitalization", "outpatient", "dental", "pharmacy"])
member_age = st.number_input("Member Age", min_value=18, max_value=90, value=45)
chronic_conditions = st.number_input("Chronic Conditions", min_value=0, max_value=5, value=1)
length_of_stay = st.number_input("Length of Stay (days)", min_value=0, max_value=30, value=0)
num_procedures = st.number_input("Number of Procedures", min_value=1, max_value=15, value=2)
procedure_category = st.selectbox("Procedure Category", ["surgery", "imaging", "lab", "consultation", "therapy"])
provider_specialty = st.selectbox("Provider Specialty", ["general", "cardiology", "orthopedics", "dentistry", "radiology"])
days_since_policy = st.number_input("Days Since Policy Start", min_value=1, max_value=730, value=100)
weekend_claim = st.checkbox("Weekend Claim")
multiple_same_day = st.checkbox("Multiple Claims Same Day")

if st.button("Check Claim"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'claim_amount': [claim_amount],
        'claim_type': [claim_type],
        'member_age': [member_age],
        'chronic_conditions_count': [chronic_conditions],
        'length_of_stay_days': [length_of_stay],
        'num_procedures': [num_procedures],
        'procedure_category': [procedure_category],
        'provider_specialty': [provider_specialty],
        'days_since_policy_start': [days_since_policy],
        'weekend_claim_flag': [1 if weekend_claim else 0],
        'multiple_claims_same_day': [1 if multiple_same_day else 0],
        'amount_per_day_of_stay': [claim_amount / max(length_of_stay, 1)],
        'cost_per_procedure': [claim_amount / num_procedures],
        'high_amount_flag': [1 if claim_amount > 15000 else 0],
        'high_cost_per_procedure': [1 if (claim_amount/num_procedures) > 5000 else 0],
        'rushed_claim': [1 if (days_since_policy <= 30 and claim_amount > 10000) else 0]
    })
    
    # Load saved preprocessing objects
    with open(f'{output_dir}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{output_dir}/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open(f'{output_dir}/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open(f'{output_dir}/isolation_forest.pkl', 'rb') as f:
        iso_model = pickle.load(f)
    with open(f'{output_dir}/lof.pkl', 'rb') as f:
        lof_model = pickle.load(f)
    
    # Preprocess
    categorical_cols = ['claim_type', 'procedure_category', 'provider_specialty']
    for col in categorical_cols:
        input_data[col] = label_encoders[col].transform(input_data[col])
    
    X_scaled = scaler.transform(input_data)
    X_pca = pca.transform(X_scaled)
    
    # Predict
    iso_score = iso_model.decision_function(X_pca)[0]
    iso_pred = iso_model.predict(X_pca)[0]
    lof_score = lof_model.score_samples(X_pca)[0]
    lof_pred = lof_model.predict(X_pca)[0]
    
    # Display results
    st.subheader("Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Isolation Forest", 
                  "🚨 SUSPICIOUS" if iso_pred == -1 else "✅ Normal",
                  f"Score: {iso_score:.4f}")
    
    with col2:
        st.metric("LOF", 
                  "🚨 SUSPICIOUS" if lof_pred == -1 else "✅ Normal",
                  f"Score: {lof_score:.4f}")
    
    # Final verdict
    if iso_pred == -1 or lof_pred == -1:
        st.error("⚠️ This claim is flagged as SUSPICIOUS - Recommend investigation")
    else:
        st.success("✓ This claim appears NORMAL")
