import streamlit as st
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# 🌈 Page setup
st.set_page_config(
    page_title="🎯 Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

# 🎨 Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #E3F2FD 0%, #FCE4EC 100%);
    }
    .stButton>button {
        background-color: #7B1FA2;
        color: white;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #9C27B0;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# 🏷️ Header
st.markdown("<h1 style='text-align:center; color:#4A148C;'>💡 Customer Churn Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6A1B9A;'>Predict whether a customer will stay 💚 or churn 💔 using AI!</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Load & preprocess dataset ---
data = pd.read_csv("customer_churn_dataset-testing-master.csv")

le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# Remove CustomerID if exists
if "CustomerID" in data.columns:
    data = data.drop("CustomerID", axis=1)

X = data.drop("Churn", axis=1)
y = data["Churn"]

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --- Input section ---
st.markdown("### 🧮 Enter Customer Details")

age = st.slider("🎂 Age", 18, 100, 30)
gender = st.radio("👫 Gender", ["Male", "Female"], horizontal=True)
tenure = st.selectbox("🕒 Tenure", ["Weekly", "Monthly", "Yearly"])
usage = st.number_input("📱 Usage Frequency", min_value=0, max_value=1000, value=100, step=1)
support_calls = st.number_input("☎️ Support Calls", min_value=0, max_value=50, value=2, step=1)
payment_delay = st.number_input("⏰ Payment Delay (days)", min_value=0, max_value=100, value=5, step=1)
subscription_type = st.selectbox("💼 Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.slider("📅 Contract Length (months)", 1, 12, 6)
total_spend = st.number_input("💸 Total Spend ($)", min_value=0, max_value=100000, value=2000, step=100)
last_interaction = st.slider("💬 Last Interaction (days ago)", 0, 365, 30)

# --- Prepare data ---
gender_val = 1 if gender == "Male" else 0
tenure_map = {"Weekly": 1, "Monthly": 2, "Yearly": 3}
subscription_map = {"Basic": 1, "Standard": 2, "Premium": 3}

input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender_val,
    "Tenure": tenure_map[tenure],
    "Usage Frequency": usage,
    "Support Calls": support_calls,
    "Payment Delay": payment_delay,
    "Subscription Type": subscription_map[subscription_type],
    "Contract Length": contract_length,
    "Total Spend": total_spend,
    "Last Interaction": last_interaction
}])

input_scaled = scaler.transform(input_data)

# --- Prediction button ---
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🚀 Predict Now!")

if predict_btn:
    with st.spinner("🔍 Analyzing data... please wait ⏳"):
        time.sleep(1.5)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        stay_prob = probabilities[0] * 100
        churn_prob = probabilities[1] * 100

    st.markdown("---")
    if prediction == 1:
        st.error("💔 The customer is **likely to CHURN!**")
        st.progress(int(churn_prob))
        st.markdown(f"### 😢 Churn Probability: **{churn_prob:.2f}%**")
        st.markdown(f"### 💚 Stay Probability: **{stay_prob:.2f}%**")
        st.warning("💡 Tip: Offer rewards or discounts to retain this customer.")
    else:
        st.balloons()
        st.success("🎉 The customer is **likely to STAY!** 💚")
        st.progress(int(stay_prob))
        st.markdown(f"### 💚 Stay Probability: **{stay_prob:.2f}%**")
        st.markdown(f"### 💔 Churn Probability: **{churn_prob:.2f}%**")
        st.info("✨ Keep up the great work! This customer seems happy!")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'></p>", unsafe_allow_html=True)
