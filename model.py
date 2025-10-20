import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- Step 1: Title ---
st.title("📊 Customer Churn Prediction System")
st.write("Use this web app to predict whether a customer will churn or stay based on their details.")

# --- Step 2: Load dataset ---
data = pd.read_csv("customer_churn_dataset-testing-master.csv")

# --- Step 3: Encode categorical columns ---
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# --- Step 4: Split features and target ---
X = data.drop("Churn", axis=1)
y = data["Churn"]

# --- Step 5: Split into train/test sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 6: Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 7: Train model ---
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --- Step 8: Evaluate model ---
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"✅ Model Trained Successfully! Accuracy: {accuracy:.2f}")

# --- Step 9: User input ---
st.header("🧮 Enter Customer Details")

user_input = {}
for col in X.columns:
    if "gender" in col.lower():
        user_input[col] = st.selectbox(col, ["Male", "Female"])
    elif "subscription" in col.lower():
        user_input[col] = st.selectbox(col, ["Basic", "Premium", "Other"])
    else:
        user_input[col] = st.number_input(f"{col}", min_value=0.0, value=0.0)

# --- Step 10: Convert input to DataFrame ---
input_df = pd.DataFrame([user_input])

# Encode categorical values the same way
for col in input_df.select_dtypes(include='object').columns:
    input_df[col] = le.fit_transform(input_df[col])

# --- Step 11: Align columns with training data ---
missing_cols = set(X.columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0  # fill missing features with 0
input_df = input_df[X.columns]  # ensure same order

# --- Step 12: Scale and predict ---
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# --- Step 13: Display result ---
if st.button("🔍 Predict Churn"):
    if prediction == 1:
        st.error(f"⚠️ The customer is likely to CHURN. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ The customer is NOT likely to churn. (Probability: {probability:.2f})")
