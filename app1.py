# 1
'''
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

# --- Load Model ---
model = joblib.load("best_fraud_model.pkl")

# --- Page Title ---
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")
st.markdown("Enter transaction details below to check for potential fraud.")

# --- Sidebar Inputs ---
st.sidebar.header("Transaction Details Input")

# You can expand or modify these input fields based on your features
def get_user_input():
    amt = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
    city_pop = st.sidebar.number_input("City Population", min_value=0.0, value=50000.0)
    unix_time = st.sidebar.number_input("Unix Time", min_value=0.0, value=1325376000.0)

    # Example categorical fields (same encoding must match model training)
    category = st.sidebar.selectbox("Category", ["shopping_pos", "gas_transport", "misc_net", "grocery_pos"])
    gender = st.sidebar.selectbox("Gender", ["M", "F"])
    merchant = st.sidebar.text_input("Merchant Name", "fraud_Kirlin and Sons")

    # Construct dictionary
    user_data = {
        "amt": amt,
        "city_pop": city_pop,
        "unix_time": unix_time,
        "category": category,
        "gender": gender,
        "merchant": merchant,
    }

    return pd.DataFrame([user_data])

input_df = get_user_input()

# --- Feature Encoding (same logic as model training) ---
def preprocess_input(df_raw):
    df = df_raw.copy()

    # Label Encoding - must match model training encoding
    # This is a simplified encoding based on training knowledge
    label_encoders = {
        "category": {'shopping_pos': 0, 'gas_transport': 1, 'misc_net': 2, 'grocery_pos': 3},
        "gender": {'M': 0, 'F': 1},
        "merchant": {'fraud_Kirlin and Sons': 0}  # Update with more as per training data
    }

    for col in ['category', 'gender', 'merchant']:
        if col in df.columns:
            df[col] = df[col].map(label_encoders.get(col, {})).fillna(0).astype(int)

    # Scale numeric fields (use same scaler logic as training)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)  # Note: fit on single row here – ideally use saved scaler

    return df_scaled

# --- Predict ---
if st.button("Predict Fraud"):
    processed_input = preprocess_input(input_df)
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1] if hasattr(model, "predict_proba") else None

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ Transaction is predicted to be **FRAUDULENT**")
    else:
        st.success("✅ Transaction is predicted to be **LEGITIMATE**")

    if prediction_proba is not None:
        st.markdown(f"**Fraud Probability:** `{prediction_proba[0]:.2%}`")
'''
# 2
'''
import streamlit as st
import pandas as pd

st.title("Fraud Detection Input Form")

with st.form("fraud_form"):
    trans_date_trans_time = st.text_input("Transaction Date & Time")
    cc_num = st.text_input("Credit Card Number")
    merchant = st.text_input("Merchant Name")
    category = st.selectbox("Category", ["shopping", "gas_transport", "grocery_pos", "misc_pos", "entertainment", "others"])
    amt = st.number_input("Amount", min_value=0.0, format="%.2f")
    first = st.text_input("First Name")
    last = st.text_input("Last Name")
    gender = st.selectbox("Gender", ["M", "F"])
    street = st.text_input("Street")
    city = st.text_input("City")
    state = st.text_input("State")
    zip_code = st.text_input("ZIP Code")
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    city_pop = st.number_input("City Population", min_value=0)
    job = st.text_input("Job")
    dob = st.text_input("Date of Birth")
    trans_num = st.text_input("Transaction Number")
    unix_time = st.number_input("Unix Time", step=1)
    merch_lat = st.number_input("Merchant Latitude")
    merch_long = st.number_input("Merchant Longitude")
    is_fraud = st.selectbox("Is Fraud?", [0, 1])

    submitted = st.form_submit_button("Submit")

if submitted:
    st.success("Form submitted successfully!")

    input_data = {
        "trans_date_trans_time": trans_date_trans_time,
        "cc_num": cc_num,
        "merchant": merchant,
        "category": category,
        "amt": amt,
        "first": first,
        "last": last,
        "gender": gender,
        "street": street,
        "city": city,
        "state": state,
        "zip": zip_code,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "job": job,
        "dob": dob,
        "trans_num": trans_num,
        "unix_time": unix_time,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "is_fraud": is_fraud
    }

    df = pd.DataFrame([input_data])
    st.write("Here's the input data:")
    st.dataframe(df)
'''
# 3
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from geopy.distance import geodesic

st.set_page_config(page_title="Fraud Detection App", layout="centered")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("best_fraud_model.pkl")

model = load_model()

st.title("🚨 Credit Card Fraud Detection App")
st.write("""
Upload your transaction CSV file. The app will preprocess the data and use a trained XGBoost model to detect fraudulent transactions.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file (no 'is_fraud' column)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📋 Uploaded Data (First 5 Rows)")
        st.dataframe(df.head())

        # =================== Data Preprocessing ===================

        # Remove 'is_fraud' if accidentally included
        df = df.drop(columns=['is_fraud'], errors='ignore')

        # Drop unnecessary columns if present
        drop_cols = ['first', 'last', 'street', 'city', 'state', 'zip', 
                     'trans_num', 'unix_time', 'merch_zipcode']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # Convert date
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        df['trans_year'] = df['trans_date_trans_time'].dt.year
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        df['trans_day'] = df['trans_date_trans_time'].dt.day
        df['trans_season'] = df['trans_date_trans_time'].dt.month % 12 // 3 + 1
        df['trans_weekday'] = df['trans_date_trans_time'].dt.weekday 
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_minute'] = df['trans_date_trans_time'].dt.minute
        df['trans_second'] = df['trans_date_trans_time'].dt.second
        df = df.drop(columns=['trans_date_trans_time'])

        # Calculate age
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['card_holder_age'] = df['trans_year'] - df['dob'].dt.year
        df = df.drop(columns=['dob'])

        # Distance between user and merchant
        def calc_distance(row):
            try:
                return geodesic((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).km
            except:
                return 0
        df['distance'] = df.apply(calc_distance, axis=1)

        # Label Encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in ['category', 'gender', 'job', 'merchant']:
            if col in df.columns:
                df[col] = le.fit_transform(df[col].astype(str))

        # Drop identifiers
        df = df.drop(columns=['cc_num'], errors='ignore')

        # Align with model's expected features
        expected_features = model.get_booster().feature_names
        missing_cols = [col for col in expected_features if col not in df.columns]
        for col in missing_cols:
            df[col] = 0  # Add missing cols as 0
        df = df[expected_features]

        # =================== Prediction ===================

        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        # Results dataframe
        result_df = pd.read_csv(uploaded_file)
        result_df["Predicted Fraud"] = preds
        result_df["Fraud Probability"] = probs

        st.subheader("✅ Prediction Results")
        st.dataframe(result_df[["cc_num", "amt", "category", "Predicted Fraud", "Fraud Probability"]])

        # Plot summary
        st.subheader("📊 Fraud Prediction Summary")
        fig, ax = plt.subplots()
        sns.countplot(x="Predicted Fraud", data=result_df, ax=ax)
        ax.set_xticklabels(["Not Fraud", "Fraud"])
        ax.set_ylabel("Number of Transactions")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Please upload a valid CSV file to begin.")
