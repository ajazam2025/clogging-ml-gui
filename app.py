
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(page_title="Clogging Prediction App")

st.title("🧱 Geopolymer Pervious Concrete – Clogging Prediction")
st.markdown(
    "Predict **Clogging Rate (% per year)** using multiple machine learning models"
)

# -----------------------------------
# Load data (from GitHub root)
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("input data.xlsx", engine="openpyxl")

df = load_data()

# -----------------------------------
# Target variable
# -----------------------------------
TARGET = "Clogging_Rate_percent_per_year"

# -----------------------------------
# REMOVE UNWANTED COLUMNS (IMPORTANT)
# -----------------------------------
DROP_COLS = [
    "NaOH_Molarity",
    "Ns_Nh_Ratio"
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# -----------------------------------
# Features and target
# -----------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# Train models (cached)
# -----------------------------------
@st.cache_resource
def train_models():

    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(C=100, gamma=0.1))
        ]),

        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            ))
        ]),

        "Bayesian Model Averaging": Pipeline([
            ("scaler", StandardScaler()),
            ("model", BayesianRidge())
        ])
    }

    r2_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_scores[name] = r2_score(y_test, y_pred)

    return models, r2_scores


with st.spinner("🔄 Training ML models..."):
    models, r2_scores = train_models()

st.success("✅ Models trained successfully")

# -----------------------------------
# Show model performance
# -----------------------------------
st.subheader("📊 Model Performance (R²)")

r2_df = pd.DataFrame.from_dict(
    r2_scores, orient="index", columns=["R² Score"]
).sort_values(by="R² Score", ascending=False)

st.dataframe(r2_df)

# -----------------------------------
# User inputs (ONLY remaining features)
# -----------------------------------
st.sidebar.header("🔧 Input Parameters")

input_data = {}

for col in X.columns:
    input_data[col] = st.sidebar.number_input(
        col,
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([input_data])

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("🚀 Predict Clogging Rate"):

    predictions = {}

    for name, model in models.items():
        predictions[name] = model.predict(input_df)[0]

    st.subheader("🔍 Predicted Clogging Rate (% per year)")

    for name, value in predictions.items():
        st.metric(name, f"{value:.2f}")

    st.success("✅ Prediction completed")
