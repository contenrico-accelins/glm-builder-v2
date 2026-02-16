import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.api import families
from statsmodels.genmod.families import Tweedie

from io import BytesIO
from utils import (
    enforce_categories,
    summarize_fit,
    residual_plot,
    format_pricing_formula,
    show_distribution,
    show_predictor_selector_sidebar,
    check_if_model_inputs_changed,
    save_model_input_snapshot,
    create_excel_report
)

st.set_page_config(page_title="GLM Builder - GLM Fit", layout="centered", page_icon="📈")
st.title("GLM Fit")

# --- Validate session state ---
if st.session_state.get('df') is None or not st.session_state.get('selected_preds'):
    st.error("⚠️ Upload your dataset and select predictors on the Home page first.")
    st.stop()

# --- Load from session ---
df_raw     = st.session_state.df_raw
df         = st.session_state.df
col_map    = st.session_state.col_map
response   = st.session_state.response
response_orig = st.session_state.response_orig
orig_preds = st.session_state.selected_preds
san_preds  = st.session_state.preds
dist       = st.session_state.dist
var_power  = st.session_state.var_power

# Use reordered dataframe if available
if 'df_reordered' in st.session_state:
    df = st.session_state.df_reordered

show_distribution()
show_predictor_selector_sidebar()

# --- View selected predictors ---
with st.expander("View selected predictors"):
    st.dataframe(pd.DataFrame({"Selected Predictors": orig_preds}), hide_index=True)

# --- Initialize exposure column ---
if "exposure_col" not in st.session_state:
    st.session_state["exposure_col"] = "None"

# --- Exposure offset selection ---
exposure_options = ["None"] + list(df.columns)
exposure_col = st.selectbox(
    "Exposure Offset (optional)",
    options=exposure_options,
    index=exposure_options.index(st.session_state["exposure_col"]) if st.session_state["exposure_col"] in df.columns else 0,
    help="Select 'None' if not using an exposure offset"
)
st.session_state["exposure_col"] = exposure_col

# --- Target Base Rate (only show if exposure offset is selected) ---
if exposure_col != "None":
    # Initialize target base rate checkbox state
    if "use_target_base_rate" not in st.session_state:
        st.session_state["use_target_base_rate"] = False
    
    use_target_base_rate = st.checkbox(
        "Target Base Rate",
        value=st.session_state["use_target_base_rate"],
        help="Use a target base rate for model fitting"
    )
    st.session_state["use_target_base_rate"] = use_target_base_rate
    
    if use_target_base_rate:
        # Initialize target base rate value
        if "target_base_rate" not in st.session_state:
            st.session_state["target_base_rate"] = 1.0
            
        target_base_rate = st.number_input(
            "Base Rate",
            min_value=0.001,
            value=st.session_state["target_base_rate"],
            step=0.1,
            format="%.3f",
            help="Target base rate for model fitting"
        )
        st.session_state["target_base_rate"] = target_base_rate
else:
    # Reset target base rate settings if no exposure offset
    st.session_state["use_target_base_rate"] = False

# --- Hold-out config ---
holdout_frac_ = st.session_state.get("holdout_frac", 0.2)
holdout_frac = st.slider("Hold-out sample %", min_value=0.05, max_value=0.5, value=holdout_frac_, step=0.05)
st.session_state["holdout_frac"] = holdout_frac

# --- Detect config changes ---
current_inputs = {
    "response": response_orig,
    "dist": dist,
    "var_power": var_power,
    "selected_preds": orig_preds.copy(),
    "exposure_col": exposure_col,
    "use_target_base_rate": st.session_state.get("use_target_base_rate", False),
    "target_base_rate": st.session_state.get("target_base_rate", 1.0)
}
inputs_changed = check_if_model_inputs_changed(current_inputs)

# --- Fit model button ---
run_glm = st.button("Fit GLM")

# --- Load from cache if safe ---
if not run_glm and "glm_model" in st.session_state and not inputs_changed:
    model = st.session_state["glm_model"]
    df_model = st.session_state["glm_df_model"]
    df_holdout = st.session_state["glm_df_holdout"]
    df = st.session_state["glm_df_full"]
    summary_df = st.session_state["glm_summary"]
    plots = st.session_state["glm_plots"]

if run_glm:
    with st.spinner("Fitting GLM model..."):
        # 1. Split hold-out sample
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        holdout_size = int(len(df_shuffled) * holdout_frac)
        df_holdout = df_shuffled.iloc[:holdout_size].copy()
        df_model = df_shuffled.iloc[holdout_size:].copy()

        # 2. Get list of SANITIZED categorical columns
        cat_cols = [san for orig, san in zip(orig_preds, san_preds) if df_raw[orig].dtype.name in ['category', 'object']]

        # 3. Synchronize categories
        df_model, df_holdout, df = enforce_categories([df_model, df_holdout, df], cat_cols)

        # 4. Build the formula
        terms = []
        for orig, san in zip(orig_preds, san_preds):
            if df_raw[orig].dtype.name in ['category', 'object']:
                terms.append(f"C({san})")
            else:
                terms.append(san)

        formula = f"{response} ~ " + " + ".join(terms)

        # 5. Select family
        if dist == "Gamma":
            family = families.Gamma(link=families.links.log())
        else:
            family = Tweedie(var_power=var_power, link=families.links.log())

        # 6. Calculate offset with target base rate if applicable
        offset_col = None
        if exposure_col != "None":
            if st.session_state.get("use_target_base_rate", False):
                fixed_intercept = np.log(st.session_state.get("target_base_rate", 1.0))
                # Add offset column to all dataframes
                for dfx in [df_model, df_holdout, df]:
                    dfx["offset"] = dfx[exposure_col] + fixed_intercept
                offset_col = "offset"
            else:
                offset_col = exposure_col

        # 7. Fit the model
        if offset_col is not None:
            model = smf.glm(formula=formula, data=df_model, family=family, offset=df_model[offset_col]).fit()
        else:
            model = smf.glm(formula=formula, data=df_model, family=family).fit()

        # 8. Predict & residuals
        for dfx, label in [(df_model, "model"), (df_holdout, "holdout"), (df, "full")]:
            if offset_col is not None:
                dfx["predicted"] = model.predict(dfx, offset=dfx[offset_col])
            else:
                dfx["predicted"] = model.predict(dfx)

            dfx["residual"] = dfx[response] - dfx["predicted"]

        # 9. Summary stats
        summary_df = pd.DataFrame({
            "Model": summarize_fit(df_model, response, model),
            "Holdout": summarize_fit(df_holdout, response, model),
            "All": summarize_fit(df, response, model)
        })

        # 10. Residual plots
        plots = {
            "Model": residual_plot(df_model, "Model"),
            "Holdout": residual_plot(df_holdout, "Holdout"),
            "All": residual_plot(df, "All")
        }

        # 11. Cache results
        st.session_state["glm_model"] = model
        st.session_state["glm_df_model"] = df_model
        st.session_state["glm_df_holdout"] = df_holdout
        st.session_state["glm_df_full"] = df
        st.session_state["glm_summary"] = summary_df
        st.session_state["glm_plots"] = plots
        save_model_input_snapshot(current_inputs)
        # Clear the inputs changed flag since we just fitted with current inputs
        inputs_changed = False

# Only show warning if there's an existing model and inputs have changed (not on first load)
# Check this AFTER potential model fitting so we account for just-fitted models
if inputs_changed and "glm_model" in st.session_state:
    st.warning("⚠️ Model inputs have changed. Please re-fit the model to update results.")

# --- Display results ---
# Show results if model exists and either: inputs are current OR we just fitted the model
show_results = "glm_model" in st.session_state and (not inputs_changed or run_glm)
if show_results:
    st.subheader("Goodness of Fit Summary")
    st.dataframe(st.session_state["glm_summary"].style.format("{:.4f}"))

    for label, fig in st.session_state["glm_plots"].items():
        st.pyplot(fig)

    st.subheader("Relativities and Coefficients")
    format_pricing_formula(st.session_state["glm_model"].params)

    st.subheader("Download Results")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        create_excel_report(writer)

    st.download_button(
        label="📥 Download Excel Report",
        data=output.getvalue(),
        file_name="glm_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
