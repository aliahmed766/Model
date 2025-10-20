# model.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# ---------------------------
# Step 1: Load dataset
# ---------------------------
data = pd.read_csv("customer_churn_dataset-testing-master.csv")

# ---------------------------
# Step 2: Encode categorical columns
# ---------------------------
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# ---------------------------
# Step 3: Split features and target
# ---------------------------
X = data.drop("Churn", axis=1)
y = data["Churn"]

# ---------------------------
# Step 4: Split train/test
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Step 5: Scale features
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# Step 6: Train Logistic Regression model
# ---------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ---------------------------
# Step 7: Streamlit UI
# ---------------------------
st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a customer will churn based on their details.")

st.header("🧮 Enter Customer Details")

# Input fields
customer_id = st.number_input("CustomerID", min_value=1, value=1, step=1)
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=600, value=12, step=1)
usage_frequency = st.number_input("Usage Frequency", min_value=0, max_value=1000, value=100, step=1)
support_calls = st.number_input("Support Calls", min_value=0, max_value=1000, value=0, step=1)
payment_delay = st.number_input("Payment Delay", min_value=0, max_value=100, value=0, step=1)
subscription_type = st.selectbox("Subscription Type", ["Basic", "Premium", "Other"])
contract_length = st.number_input("Contract Length", min_value=0, max_value=1000, value=12, step=1)
total_spend = st.number_input("Total Spend", min_value=0, max_value=100000, value=1000, step=1)
last_interaction = st.number_input("Last Interaction (days)", min_value=0, max_value=365, value=10, step=1)

# Button to predict
if st.button("Predict Churn"):
    # Prepare input DataFrame matching training features
    input_data = pd.DataFrame({
        "CustomerID": [customer_id],
        "Age": [age],
        "Gender": [1 if gender.lower() == "male" else 0],
        "Tenure": [tenure],
        "Usage Frequency": [usage_frequency],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Subscription Type": [1 if subscription_type.lower() == "premium" else 0],
        "Contract Length": [contract_length],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Show result
    if prediction == 1:
        st.error(f"⚠️ The customer is likely to CHURN. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ The customer is NOT likely to churn. (Probability: {probability:.2f})")
