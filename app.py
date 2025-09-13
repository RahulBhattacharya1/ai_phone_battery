import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Phone Battery Estimator", page_icon="🔋", layout="centered")
st.title("Phone Battery Estimator")
st.write("Enter key specs to estimate **battery capacity (mAh)**.")

MODEL_PATH = os.path.join("models", "phone_battery_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

with st.form("specs"):
    col1, col2 = st.columns(2)

    with col1:
        os_family = st.selectbox("OS", ["android","ios","windows","other"], index=0)
        chipset_brand = st.selectbox(
            "Chipset Brand",
            ["qualcomm","mediatek","exynos","apple","unisoc","other"],
            index=0
        )
        screen_in = st.number_input("Screen Size (inches)", min_value=3.5, max_value=8.5, value=6.5, step=0.1, format="%.1f")

    with col2:
        ram_gb = st.number_input("RAM (GB)", min_value=1.0, max_value=24.0, value=8.0, step=1.0)
        storage_gb = st.number_input("Storage (GB)", min_value=8.0, max_value=1024.0, value=128.0, step=16.0)
        refresh_hz = st.number_input("Refresh Rate (Hz)", min_value=30.0, max_value=240.0, value=120.0, step=10.0)

    supports_5g = st.selectbox("5G Support", ["no","yes"], index=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    x = pd.DataFrame([{
        "os_family": os_family,
        "chipset_brand": chipset_brand,
        "ram_gb": float(ram_gb),
        "storage_gb": float(storage_gb),
        "screen_in": float(screen_in),
        "refresh_hz": float(refresh_hz),
        "supports_5g": 1 if supports_5g == "yes" else 0
    }])

    try:
        pred = float(model.predict(x)[0])
        st.success(f"Estimated Battery Capacity: **{pred:,.0f} mAh**")
        st.caption("Estimate based on a model trained from a public smartphone specs dataset.")
    except Exception as e:
        st.error("Prediction failed. Ensure the model file path is correct and matches training features.")
        st.exception(e)

with st.expander("How this works"):
    st.markdown(
        """
        **Model**: scikit-learn Pipeline (One-Hot Encoding for OS/chipset + RandomForestRegressor).
        **Input features**: OS family, chipset brand, RAM, storage, screen size, refresh rate, and a 5G flag.
        **Target**: battery capacity (mAh).  
        The model artifact is compressed so it stays under 25 MB for GitHub uploads.
        """
    )

