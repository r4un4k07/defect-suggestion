import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import differential_evolution
import time
from PIL import Image


import streamlit as st

# Set page config before any other Streamlit command
st.set_page_config(page_title="Steel Defect Prediction", layout="wide")

# --- Simple Custom Routing ---
page = st.query_params.get("page", "model")

# If user navigates to ?page=info, load info.py dynamically
if page == "info":
    import resources.info as info
    info.show()
    st.stop()
# --- Load assets ---
rf_model = joblib.load("./model_files/rf_model.pkl")
lgb_model = lgb.Booster(model_file="./model_files/lgb_model.txt")
scaler = joblib.load("./model_files/scaler.pkl")
label_encoder = joblib.load("./model_files/label_encoder.pkl")

feature_names = [
    'Carbon (C)', 'Manganese (Mn)', 'Sulfur (S)', 'Phosphorus (P)', 'Silicon (Si)',
    'Nickel (Ni)', 'Chromium (Cr)', 'Copper (Cu)', 'Titanium (Ti)', 'Cobalt (Co)',
    'Nitrogen (N)', 'Lead (Pb)', 'Tin (Sn)', 'Aluminum (Al)', 'Boron (B)',
    'Vanadium (V)', 'Calcium (Ca)', 'Niobium (Nb)'
]
short_names = [f.split()[0] for f in feature_names]

def_val = [69, 3200, 15, 310, 4000, 1600, 173500, 500, 3200,
           400, 100, 45, 50, 365, 5, 440, 10, 20]

bounds_raw = [
    (0, 100), (2000, 3500), (0, 30), (200, 500), (2500, 5200),
    (0, 3000), (112000, 174000), (0, 2000), (2000, 3500),
    (0, 500), (0, 125), (0, 45), (0, 50), (100, 500),
    (0, 50), (0, 500), (0, 200), (0, 200)
]

if "download_success_time" not in st.session_state:
    st.session_state.download_success_time = None

# Initialize session state variables
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "optimized" not in st.session_state:
    st.session_state.optimized = False
if "last_input" not in st.session_state:
    st.session_state.last_input = def_val.copy()
if "opt_df" not in st.session_state:
    st.session_state.opt_df = None
if "opt_prob" not in st.session_state:
    st.session_state.opt_prob = None
if "opt_time" not in st.session_state:
    st.session_state.opt_time = None
if "current_values" not in st.session_state:
    st.session_state.current_values = def_val.copy()
if "last_validation_time" not in st.session_state:
    st.session_state.last_validation_time = 0
if "input_changed_time" not in st.session_state:
    st.session_state.input_changed_time = 0

# --- Sidebar ---
# College Logo at the top
import streamlit as st
import base64

def load_logo_base64(path):
    with open(path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    return b64

logo_b64 = load_logo_base64("./resources/logo.png")

# Add sidebar navigation buttons



st.sidebar.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{logo_b64}" width="100" height="100" style="image-rendering: crisp-edges;"/>
    </div>
""", unsafe_allow_html=True)

model_choice = st.sidebar.selectbox("🔍 Select Model", ["Random Forest", "LightGBM"], key="model_select")

# Model description box
model_descriptions = {
    "Random Forest": "An ensemble learning method that uses multiple decision trees. Known for high accuracy, robustness to overfitting, and good performance on tabular data with feature importance insights.",
    "LightGBM": "A gradient boosting framework optimized for speed and memory efficiency. Excellent for large datasets with fast training and high accuracy, using leaf-wise tree growth."
}

with st.sidebar.container():
    st.caption(model_descriptions[model_choice])



if st.sidebar.button("ℹ️ How This Works"):
    st.query_params["page"] = "info"
    st.rerun()

#Guided by
st.sidebar.markdown("### 👥 Guided by")
st.sidebar.markdown("""
<div style="background-color: #1f2937; padding: 16px; border-radius: 12px; border: 1px solid #374151; color: #d1d5db; font-size: 14px; line-height: 1.6;">
    <div><strong style="font-size: 16px; color: #60a5fa;">Prof. Kirty Madhavi</strong></div>
    <div style="font-size: 14px;">Assistant Professor</div>
    <div style="font-size: 14px;">Department of Metallurgical Engineering</div>
    <div style="font-size: 14px;">B.I.T. Sindri</div>
</div>
""", unsafe_allow_html=True)

# Student info with dark styling
st.sidebar.markdown("### 🎓 Student Info")
st.sidebar.markdown("""
<div style="background-color: #1f2937; padding: 16px; border-radius: 12px; border: 1px solid #374151; color: #d1d5db; font-size: 14px; line-height: 1.6;">
    <div><strong style="font-size: 16px; color: #60a5fa;">Raunak Rawani</strong> <span style="font-size: 14px;">(2103036)</span></div>
    <div><strong style="font-size: 16px; color: #60a5fa;">Parvati Kumari</strong> <span style="font-size: 14px;">(2203005D)</span></div>
    <div style="margin-top: 8px;"><strong>B.Tech 8<sup>th</sup> Semester</strong></div>
    <div><strong>Metallurgical Engineering</strong></div>
    <div><strong>B.I.T. Sindri</strong></div>
</div>
""", unsafe_allow_html=True)

# Copyright in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 13px; padding: 8px;">
        <p style="margin: 0;">© 2025 Steel Defect Prediction</p>
        <p style="margin: 2px 0 0 0; font-size: 11px;">All Rights Reserved</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- Predict Function ---
def predict(model, X_scaled):
    if model_choice == "Random Forest":
        pred = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[0]
    else:
        pred = [np.argmax(model.predict(X_scaled, pred_contrib=False))]
        probs = model.predict(X_scaled, pred_contrib=False)[0]
    return pred, probs

# Function to reset prediction and optimization states
def reset_states():
    st.session_state.predicted = False
    st.session_state.optimized = False
    st.session_state.opt_df = None
    st.session_state.opt_prob = None
    st.session_state.opt_time = None

# Function to validate input bounds
def validate_inputs(user_input):
    """
    Validate if all input values are within their respective bounds.
    Returns a tuple (is_valid, out_of_bounds_features)
    """
    out_of_bounds = []
    for i, (value, (min_val, max_val)) in enumerate(zip(user_input, bounds_raw)):
        if value < min_val or value > max_val:
            out_of_bounds.append({
                'feature': feature_names[i],
                'value': value,
                'min': min_val,
                'max': max_val
            })
    
    return len(out_of_bounds) == 0, out_of_bounds

# --- Main UI ---
st.title("🔍 Steel Defect Prediction and Optimization")
st.markdown("Enter the chemical composition to predict the defect type and get suggestions to improve quality.")
st.caption("All the values are in ppm (parts per million).")

cols = st.columns(6)
user_input = []
for i, (feature, short) in enumerate(zip(feature_names, short_names)):
    with cols[i % 6]:
        min_val, max_val = bounds_raw[i]
        val = st.number_input(
            feature, 
            value=st.session_state.current_values[i], 
            key=f"{short}_input",
            help=f"Valid range: {min_val} - {max_val} ppm"
        )
        user_input.append(val)

user_array = np.array(user_input).reshape(1, -1)

# Check if input has changed to reset states
current_time = time.time()
if not np.array_equal(user_input, st.session_state.last_input):
    reset_states()
    st.session_state.last_input = user_input.copy()
    st.session_state.current_values = user_input.copy()
    st.session_state.input_changed_time = current_time

# Only validate after a delay (1 second) from last input change
should_validate = (current_time - st.session_state.input_changed_time) >= 1.0
is_valid = True
out_of_bounds_features = []

if should_validate:
    # Validate inputs
    is_valid, out_of_bounds_features = validate_inputs(user_input)
    st.session_state.last_validation_time = current_time
elif st.session_state.input_changed_time > 0:
    # Auto-refresh after 1 second to trigger validation
    remaining_time = 1.0 - (current_time - st.session_state.input_changed_time)
    if remaining_time > 0:
        time.sleep(remaining_time)
        st.rerun()

# Display validation warnings
if not is_valid:
    st.markdown("### ⚠️ Input Validation Warnings")
    
    warning_message = "**The following values are out of bounds:**\n\n"
    for feature_info in out_of_bounds_features:
        warning_message += f"\n• **{feature_info['feature']}**: {feature_info['value']} ppm (Valid range: {feature_info['min']} - {feature_info['max']} ppm)\n"
    
    st.warning(warning_message)
    
    # Show bounds table for reference
    bounds_df = pd.DataFrame({
        'Element': feature_names,
        'Minimum (ppm)': [bound[0] for bound in bounds_raw],
        'Maximum (ppm)': [bound[1] for bound in bounds_raw],
        'Current Value (ppm)': user_input,
        'Status': ['✅ Valid' if bounds_raw[i][0] <= user_input[i] <= bounds_raw[i][1] else '❌ Out of bounds' 
                  for i in range(len(user_input))]
    })
    
    st.markdown("### 📊 Input Validation Summary")
    st.dataframe(
        bounds_df.style.apply(
            lambda row: ['background-color: #ffcdd2; color: #c62828; font-weight: bold' if row['Status'] == '❌ Out of bounds' else '' for _ in row], 
            axis=1
        ).format({
            'Minimum (ppm)': '{:.0f}',
            'Maximum (ppm)': '{:.0f}',
            'Current Value (ppm)': '{:.2f}'
        }),
        use_container_width=True
    )

user_scaled = scaler.transform(user_array)

# Predict button - disabled when inputs are invalid
predict_button = st.button(
    "🔍 Test Prediction", 
    disabled=not is_valid,
    help="Please correct the out-of-bounds values to enable prediction" if not is_valid else "Click to predict steel defect"
)

if predict_button and is_valid:
    st.session_state.predicted = True
    st.session_state.optimized = False
    st.session_state.opt_df = None
    st.session_state.opt_prob = None
    st.session_state.opt_time = None

if st.session_state.predicted and is_valid:
    model = rf_model if model_choice == "Random Forest" else lgb_model
    pred_label_idx, class_probs = predict(model, user_scaled)
    pred_label = label_encoder.inverse_transform(pred_label_idx)[0]

    inclusion_free_idx = list(label_encoder.classes_).index("Inclusion Free")
    inclusion_free_prob = class_probs[inclusion_free_idx]

    if pred_label == "Inclusion Free":
        st.success(f"✅ The material is predicted to be **Inclusion Free** with a probability of {inclusion_free_prob:.4f}.")
    else:
        st.warning(f"⚠️ Possible Defect: **{pred_label}**")

    prob_df = pd.DataFrame({
        "Defect Type": label_encoder.classes_,
        "Probability": class_probs
    }).sort_values("Probability", ascending=False).head(3)

    st.markdown("### 🔢 Top 3 Probable Classes")
    st.dataframe(
        prob_df.style
            .format({"Probability": "{:.4f}"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]}]),
        use_container_width=True
    )

    if not st.session_state.optimized:
        if inclusion_free_prob < 0.8:  # Only show optimize button if probability is less than 0.8
            if st.button("🧪 Optimize for 'Inclusion Free'"):
                st.session_state.optimized = True

if st.session_state.optimized and st.session_state.opt_df is None and is_valid:
    scaled_bounds = []
    for i, (low, high) in enumerate(bounds_raw):
        x_low = user_array[0].copy()
        x_high = user_array[0].copy()
        x_low[i] = low
        x_high[i] = high
        scaled_low = scaler.transform([x_low])[0][i]
        scaled_high = scaler.transform([x_high])[0][i]
        scaled_bounds.append((scaled_low, scaled_high))

    model = rf_model if model_choice == "Random Forest" else lgb_model

    def objective(x):
        if model_choice == "Random Forest":
            return -model.predict_proba([x])[0][inclusion_free_idx]
        else:
            proba = model.predict([x], pred_contrib=False)
            return -proba[0][inclusion_free_idx]

    start_time = time.time()
    with st.spinner("🕐 Optimizing... Please wait."):
        result = differential_evolution(objective, scaled_bounds, maxiter=1000, polish=True, seed=42)
    end_time = time.time()

    elapsed = end_time - start_time
    st.session_state.opt_time = elapsed

    opt_scaled = result.x
    opt_input = scaler.inverse_transform([opt_scaled])[0]
    opt_prob = -result.fun
    delta = opt_input - user_array[0]

    st.session_state.opt_df = pd.DataFrame({
        "Element": feature_names,
        "Original (ppm)": user_array[0],
        "Suggested (ppm)": opt_input,
        "Change (Δ ppm)": delta
    })
    st.session_state.opt_prob = opt_prob

if st.session_state.opt_df is not None and is_valid:
    # Display optimization results with improved UI
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%); 
                    padding: 20px; 
                    border-radius: 12px; 
                    border-left: 5px solid #28a745; 
                    margin: 25px 0; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 28px; margin-right: 12px;">🎯</span>
                <h3 style="color: #155724; margin: 0; font-size: 22px; font-weight: 600;">
                    Optimization Results
                </h3>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.7); 
                           padding: 12px; 
                           border-radius: 8px; 
                           border: 1px solid rgba(40,167,69,0.2);">
                    <div style="font-size: 14px; color: #6c757d; margin-bottom: 4px;">
                        Inclusion Free Probability
                    </div>
                    <div style="font-size: 24px; font-weight: 700; color: #28a745;">
                        {st.session_state.opt_prob:.4f}
                    </div>
                    <div style="font-size: 12px; color: #28a745; margin-top: 2px;">
                        ({st.session_state.opt_prob*100:.2f}%)
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.7); 
                           padding: 12px; 
                           border-radius: 8px; 
                           border: 1px solid rgba(40,167,69,0.2);">
                    <div style="font-size: 14px; color: #6c757d; margin-bottom: 4px;">
                        Optimization Time
                    </div>
                    <div style="font-size: 24px; font-weight: 700; color: #28a745;">
                        {st.session_state.opt_time:.2f}s
                    </div>
                    <div style="font-size: 12px; color: #6c757d; margin-top: 2px;">
                        Processing Time
                    </div>
                </div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown("### 🔧 Suggested Adjustments")
    st.dataframe(
        st.session_state.opt_df.style
            .format({
                "Original (ppm)": "{:.2f}",
                "Suggested (ppm)": "{:.2f}",
                "Change (Δ ppm)": "{:+.2f}"
            })
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]}]),
        use_container_width=True
    )
    
    # Add the Fill Optimized Values button
    if st.button("🔄 Fill Optimized Values"):
        # Update current values with optimized values
        optimized_values = st.session_state.opt_df["Suggested (ppm)"].tolist()
        st.session_state.current_values = optimized_values
        
        # Reset prediction and optimization states
        reset_states()
        
        # Force rerun to update the input fields
        st.rerun()