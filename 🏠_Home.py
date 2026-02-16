from icecream import ic
import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    sanitize_name,
    clean_dataframe, 
    percent_usable, 
    get_variable_type, 
    plot_response_distribution, 
    optimize_tweedie
)
import statsmodels.formula.api as smf

st.set_page_config(page_title="GLM Builder - Home", layout="centered", page_icon="🏠")
st.title("GLM Builder")

# 1) Upload or retrieve dataset
uploaded = st.file_uploader("Upload your insurance CSV dataset", type=["csv"])

if uploaded:
    st.session_state["uploaded_file"] = uploaded
elif "uploaded_file" in st.session_state:
    uploaded = st.session_state["uploaded_file"]

if uploaded is None and "df" not in st.session_state:
    st.info("Please upload a CSV file to proceed with analysis.")
    st.stop()

# 2) Read & sanitize if freshly uploaded
# This block runs only once per upload
if uploaded and "df" not in st.session_state:
    df_raw = pd.read_csv(uploaded)
    df_raw = clean_dataframe(df_raw)
    col_map = {col: sanitize_name(col) for col in df_raw.columns}
    df = df_raw.rename(columns=col_map)

    st.session_state['df_raw']  = df_raw
    st.session_state['df']      = df
    st.session_state['col_map'] = col_map
    
    # Clear old selections on new file upload
    for key in ['predictor_selector_state', 'selected_preds']:
        if key in st.session_state:
            del st.session_state[key]


# 3) Load from session state
df_raw = st.session_state['df_raw']
df     = st.session_state['df']
col_map= st.session_state['col_map']

response_orig = st.session_state.get("response_orig")
predictor_selector_state = st.session_state.get("predictor_selector_state")

# 4) Data preview
st.subheader("Data Preview")
st.dataframe(df_raw.head(10), hide_index=True)

# 5) Select response variable
orig_cols = list(df_raw.columns)

# Try to match first column that contains 'premium' (case-insensitive)
if response_orig in orig_cols:
    default_idx = orig_cols.index(response_orig)
else:
    match_idx = next(
        (i for i, col in enumerate(orig_cols) if "premium" in col.lower()),
        0  # fallback if no match
    )
    default_idx = match_idx

response_orig = st.selectbox("Select response variable", orig_cols, index=default_idx)

# 6) Build predictor selection table
candidate_predictors = [c for c in orig_cols if c != response_orig]



types = [get_variable_type(df_raw[c]) for c in candidate_predictors]
usable = [percent_usable(df_raw[c]) for c in candidate_predictors]

# Use existing state for flags, otherwise default to False
predictor_flags = (
    predictor_selector_state
    if predictor_selector_state and len(predictor_selector_state) == len(candidate_predictors)
    else [False] * len(candidate_predictors)
)

pred_df = pd.DataFrame({
    "variable": candidate_predictors,
    "type": types,
    "% usable": usable,
    "use_as_predictor": predictor_flags
})

# Use a form to batch the predictor selections and prevent re-renders
with st.form(key="predictor_form"):
    st.subheader("Select predictors")
    edited = st.data_editor(
        data=pred_df,
        column_config={
            "use_as_predictor": st.column_config.CheckboxColumn(
                "Predictor?",
                help="Tick variables to include as predictors",
                default=False,
                width="small"
            )
        },
        disabled=["variable", "type", "% usable"],
        hide_index=True,
        num_rows="fixed"
    )

    submitted = st.form_submit_button("Confirm Predictors & Selections")

# 7) Process selections and update state only AFTER the form is submitted
if submitted:
    selected_preds = edited.loc[edited["use_as_predictor"], "variable"].tolist()
    
    # 9) Store all selections in session state
    st.session_state['response_orig']           = response_orig
    st.session_state['response']               = sanitize_name(response_orig)
    st.session_state['selected_preds']         = selected_preds
    st.session_state['predictor_selector_state']= edited["use_as_predictor"].tolist()
    st.session_state['preds']                  = [sanitize_name(p) for p in selected_preds]
    
    # Force a re-run to cleanly reflect the new state from the top of the script
    st.rerun()

# This part now reads from the stable session state after submission
if "selected_preds" in st.session_state and st.session_state["selected_preds"]:
    st.write("Selected Predictors")
    st.dataframe(pd.DataFrame({"Selected Predictors": st.session_state["selected_preds"]}), hide_index=True)
else:
    st.warning("Please select at least one predictor and click 'Confirm'.")
    st.stop() # Stop the script here if predictors aren't confirmed


# Retrieve latest selections from state for downstream use
response = st.session_state['response']
preds    = st.session_state['preds']

# 8) Choose GLM distribution
distributions = ["Gamma", "Gaussian", "Poisson", "Tweedie"]
dist = st.selectbox(
    "Select GLM distribution",
    distributions,
    index=0 if "dist" not in st.session_state else distributions.index(st.session_state.get("dist", "Gamma"))
)
var_power = None
if dist == "Tweedie":
    var_power = st.slider("Tweedie variance power", 1.0, 2.0, st.session_state.get("var_power", 1.5))

    # 8.5) Optimize Tweedie power
    if st.checkbox("📊 Check optimal Tweedie variance power (β)"):
        optimize_tweedie(st.session_state['selected_preds'], df_raw, df, response_orig)


# 9) Final state update for distribution choice
st.session_state['dist']      = dist
st.session_state['var_power'] = var_power

# 10) Visualize selected response
st.subheader(f"Distribution of '{response_orig}'")
plot_response_distribution(df, response)