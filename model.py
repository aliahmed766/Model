# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 🎨 Streamlit page setup
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction System")
st.markdown("Use this web app to predict whether a customer will **churn** or **stay** based on their details.")

# 🔹 Step 1: Load the dataset
data = pd.read_csv("customer_churn_dataset-testing-master.csv")

# 🔹 Step 2: Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# 🔹 Step 3: Split data
X = data.drop("Churn", axis=1)
y = data["Churn"]

# 🔹 Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 5: Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Step 6: Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 🔹 Step 7: Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
st.success(f"✅ Model Trained Successfully! Accuracy: {accuracy:.2f}")

st.markdown("---")

# 🔹 Step 8: Input section
st.header("🧮 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=1000, value=24)
    monthly_usage = st.number_input("Monthly Usage", min_value=0, max_value=1000, value=200)
    total_spend = st.number_input("Total Spend", min_value=0, max_value=100000, value=10000)

with col2:
    contract_length = st.number_input("Contract Length (months)", min_value=1, max_value=60, value=12)
    gender = st.selectbox("Gender", ["Male", "Female"])
    last_interaction = st.number_input("Last Interaction (days)", min_value=0, max_value=365, value=30)
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Premium", "Other"])

# 🔹 Step 9: Predict button
if st.button("🔍 Predict Churn"):
    # Create DataFrame for input
    input_data = pd.DataFrame({
        "Age": [age],
        "Tenure": [tenure],
        "Monthly Usage": [monthly_usage],
        "Total Spend": [total_spend],
        "Contract Length": [contract_length],
        "Gender": [1 if gender == "Male" else 0],
        "Last Interaction": [last_interaction],
        "Subscription Type": [1 if subscription_type == "Premium" else 0]
    })

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to CHURN** (Probability: {probability:.2f})")
    else:
        st.success(f"✅ The customer is **NOT likely to churn** (Probability: {probability:.2f})")
