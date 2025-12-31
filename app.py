import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Startup Funding Predictor",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 Startup Funding Prediction App")

st.info(
    "This application predicts the **expected funding amount (in Indian Rupees Crore)** "
    "for a startup using **Machine Learning regression models**."
)

data = pd.read_csv("startup_funding_100_inr.csv")
data.columns = data.columns.str.strip()

le_sector = LabelEncoder()
le_location = LabelEncoder()
le_product = LabelEncoder()

data["Sector"] = le_sector.fit_transform(data["Sector"])
data["Location"] = le_location.fit_transform(data["Location"])
data["ProductType"] = le_product.fit_transform(data["ProductType"])

target_column = "FundingAmount_INR_Crore"

X = data.drop(target_column, axis=1)
y = data[target_column]

feature_columns = X.columns.tolist()

lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

st.subheader("📋 Enter Startup Details")

st.write("Fill in the startup information below to estimate the funding amount.")

sector = st.selectbox("Startup Sector", le_sector.classes_)
location = st.selectbox("Startup Location", le_location.classes_)
product_type = st.selectbox("Product Type", le_product.classes_)

team_size = st.number_input("Team Size (Employees)", min_value=1, value=10)
founder_experience = st.number_input("Founders' Experience (Years)", min_value=0, value=5)
previous_rounds = st.number_input("Previous Funding Rounds", min_value=0, value=1)

input_data = pd.DataFrame([{
    "Sector": le_sector.transform([sector])[0],
    "TeamSize": team_size,
    "FounderExperience": founder_experience,
    "Location": le_location.transform([location])[0],
    "PreviousFundingRounds": previous_rounds,
    "ProductType": le_product.transform([product_type])[0]
}], columns=feature_columns)

st.divider()

if st.button("🔮 Predict Funding Amount"):
    prediction = rf.predict(input_data)[0]
    st.success(f"💰 Estimated Funding: ₹ {prediction:.2f} Crore")

st.divider()

st.subheader("📊 Important Factors Affecting Funding")

st.write(
    "The following table shows the factors identified by **Lasso Regression** "
    "that influence startup funding."
)

importance = pd.Series(lasso.coef_, index=feature_columns)
st.dataframe(importance, use_container_width=True)
