import streamlit as st
from model import load_model


st.set_page_config(page_title="Delivery Time Predictor", page_icon="🚚")
st.title("Delivery Time Predictor")
st.write("A small Random Forest model for delivery-time estimation.")

distance = st.slider("Distance (km)", 1.0, 30.0, 8.0, 0.5)
weight = st.slider("Package weight (kg)", 0.2, 20.0, 2.0, 0.1)
priority = st.selectbox("Priority", ["Standard", "Express"])

if st.button("Predict delivery time", type="primary"):
    model = load_model()
    value = model.predict([[distance, weight, int(priority == "Express")]])[0]
    st.success(f"Estimated delivery time: {value:.1f} minutes")
