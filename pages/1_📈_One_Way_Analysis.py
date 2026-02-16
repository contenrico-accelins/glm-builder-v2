import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from utils import (
    one_way_analysis,
    plot_predictor_effect,
    show_distribution,
    show_predictor_selector_sidebar,
    check_if_model_inputs_changed,
    save_model_input_snapshot,
    create_category_csv_download,
    upload_category_csv_and_preview,
    show_category_reorder_preview,
    apply_category_reordering_confirmed
)

# --- Validate session state ---
if st.session_state.get('df') is None:
    st.error("⚠️ Upload your CSV on the Home page first.")
    st.stop()

if not st.session_state.get('selected_preds'):
    st.warning("⚠️ Select at least one predictor on the Home page.")
    st.stop()

# --- Pull from session ---
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

st.title("One‑Way Analysis")

# --- View selected predictors ---
with st.expander("View selected predictors"):
    st.dataframe(
        pd.DataFrame({"Selected Predictors": orig_preds}),
        hide_index=True
    )

# --- Detect input changes ---
current_inputs = {
    "response": response_orig,
    "dist": dist,
    "var_power": var_power,
    "selected_preds": orig_preds.copy(),
    "exposure_col": st.session_state.get("exposure_col", "None")
}

# Check if substantive model inputs changed 
inputs_changed = check_if_model_inputs_changed(current_inputs)

# --- Run analysis button ---
run_analysis = st.button("Run One-Way Analysis")

# --- Trigger or reuse analysis
if run_analysis or ("one_way_summary" in st.session_state and not inputs_changed):
    if run_analysis or inputs_changed:
        with st.spinner("Running one-way analysis..."):
            summary = one_way_analysis(df, response, san_preds, dist, var_power)
            inv_map = {v: k for k, v in col_map.items()}
            summary['Predictor'] = summary['Predictor'].map(inv_map)
            st.session_state["one_way_summary"] = summary
            save_model_input_snapshot(current_inputs)
            # Clear the inputs changed flag since we just ran with current inputs
            inputs_changed = False
    else:
        summary = st.session_state["one_way_summary"]

    # Only show warning if there's an existing analysis and inputs have changed (not on first load)
    # Check this AFTER potential analysis so we account for just-run analysis
    if inputs_changed and "one_way_summary" in st.session_state:
        st.warning("⚠️ Model inputs have changed. Please re-run the analysis.")

    # --- Summary table ---
    st.subheader("Summary Table")
    st.dataframe(
        summary.style.format({
            'p-value': '{:.3f}',
            'AIC': '{:.1f}',
            'BIC': '{:.1f}',
            'Deviance Explained': '{:.3f}'
        })
    )

    # --- Bar chart ---
    st.subheader("Deviance Explained by Predictor")
    fig, ax = plt.subplots()
    summary.plot.bar(x='Predictor', y='Deviance Explained', legend=False, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel("Deviance Explained")
    st.pyplot(fig)

    # --- Category Reordering ---
    st.subheader("Category Reordering")
    st.markdown("Reorder categories for categorical predictors to control the display order in plots.")
    
    # Check if reordering was just confirmed
    if st.session_state.get('reordering_just_confirmed', False):
        st.success("✅ Category reordering applied successfully!")
        # Clear the confirmation flag
        st.session_state['reordering_just_confirmed'] = False
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Download Current Order**")
            category_df_current = create_category_csv_download(df, san_preds)
        
        with col2:
            st.markdown("**Upload New Order**")
            uploaded_category_df = upload_category_csv_and_preview(df, san_preds)
        
        # Show preview and confirmation if CSV was uploaded
        if uploaded_category_df is not None:
            if show_category_reorder_preview(df, uploaded_category_df, san_preds):
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("✅ Confirm Reordering", type="primary"):
                        df_reordered = apply_category_reordering_confirmed(df, uploaded_category_df)
                        df = df_reordered
                        st.session_state['df_reordered'] = df_reordered
                        st.session_state['reordering_just_confirmed'] = True
                        # Update the model input snapshot since category reordering doesn't change core inputs
                        save_model_input_snapshot(current_inputs)
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel"):
                        st.rerun()

    # --- Individual plots ---
    st.subheader("Individual Predictor Effects")
    col1, col2 = st.columns(2)
    with col1:
        first_san = st.selectbox("Predictor", san_preds, index=0)
    with col2:
        st.text(" ")
        st.text(" ")
        show_all_plots = st.toggle("Show all plots")

    if show_all_plots:
        for pred in san_preds:
            plot_predictor_effect(df, response, pred)
    else:
        plot_predictor_effect(df, response, first_san)

else:
    st.info("Click the button above to run one-way analysis.")
