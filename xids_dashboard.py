import streamlit as st

st.set_page_config(page_title="X-IDS Dashboard Demo", layout="wide")
st.title("🛡️ X-IDS Dashboard Demo")

# -------------------------------
# Fixed Demo Prediction
# -------------------------------
st.subheader("Sample Predicted Attack")
st.write("**Predicted Attack:** dos")

st.subheader("Top Contributing Features")
st.write("""
- src_bytes = 0.0 → SHAP value: -0.022  
- duration = 0.0 → SHAP value: 0.012  
- protocol_type = 0.5 → SHAP value: 0.012
""")
