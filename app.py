import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model and preprocessing pipeline
with open('smote_lgbm.pkl', 'rb') as file:
    model = pickle.load(file)

with open('preprocessing_pipeline.pkl', 'rb') as file:
    preprocessing_pipeline = pickle.load(file)

# Streamlit UI
st.title("Customer Churn Prediction App")
st.subheader("Predict the likelihood of customer churn based on key features.")

# Input fields for only the selected features
col1, col2 = st.columns(2)

with col1:
    account_age = st.number_input("Account Age (months)", min_value=1, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=600.0)
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    viewing_hours_per_week = st.number_input("Viewing Hours Per Week", min_value=0.0, max_value=100.0, value=10.5)

with col2:
    average_viewing_duration = st.number_input("Average Viewing Duration (hours)", min_value=0.0, max_value=10.0, value=2.5)
    content_downloads_per_month = st.number_input("Content Downloads Per Month", min_value=0, max_value=50, value=5)
    user_rating = st.number_input("User Rating (1-5)", min_value=1.0, max_value=5.0, value=4.5)
    support_tickets_per_month = st.number_input("Support Tickets Per Month", min_value=0, max_value=10, value=1)
    watchlist_size = st.number_input("Watchlist Size", min_value=0, max_value=100, value=8)

# Collect the input into a dictionary
user_input = {
    "AccountAge": account_age,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "SubscriptionType": subscription_type,
    "ViewingHoursPerWeek": viewing_hours_per_week,
    "AverageViewingDuration": average_viewing_duration,
    "ContentDownloadsPerMonth": content_downloads_per_month,
    "UserRating": user_rating,
    "SupportTicketsPerMonth": support_tickets_per_month,
    "WatchlistSize": watchlist_size
}

user_input_df = pd.DataFrame([user_input])
user_input_processed = preprocessing_pipeline.transform(user_input_df)

# Predict churn and probability
prediction = model.predict(user_input_processed)
probability = model.predict_proba(user_input_processed)[:, 1]

# Display the results
if st.button("Predict Churn"):
    st.write("Prediction", "The customer is likely to churn, indicating a higher chance of subscription cancellation" if prediction[0] == 1  else "The customer is unlikely to churn and seems likely to continue their subscription.")
    st.write(f"Churn confidence obtained from the model: {probability[0]:.2f}")
