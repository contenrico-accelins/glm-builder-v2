import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_glm_from_disk, save_glm_to_disk, get_ai_insights

st.set_page_config(page_title="GLM Builder - Model Manager", layout="centered", page_icon="🗂️")
st.title("📁 Model Manager")

Path("saved_models").mkdir(exist_ok=True)

# --- Save current model ---
if "glm_model" in st.session_state:
    with st.expander("💾 Save Current Fitted Model"):
        model_name = st.text_input("Model name", value="model_1")
        if st.button("Save Model"):
            save_glm_to_disk(name=model_name)
            st.success(f"Model '{model_name}' saved to disk.")
else:
    st.info("Fit a model on the GLM Fit page before saving.")

# --- List saved models ---
model_files = [f for f in os.listdir("saved_models") if f.endswith(".pkl")]

if model_files:
    with st.expander("📂 Load a Saved Model"):
        selected_model = st.selectbox("Choose a saved model", model_files, format_func=lambda x: x.replace(".pkl", ""))
        if st.button("Load Selected Model"):
            load_glm_from_disk(selected_model.replace(".pkl", ""))
            st.success(f"Model '{selected_model}' loaded into session.")

    with st.expander("🧾 Model Info Preview"):
        model_path = f"saved_models/{selected_model}"
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        st.markdown(f"**Name:** `{selected_model.replace('.pkl', '')}`")
        st.markdown(f"**Response:** `{bundle['config']['response']}`")
        st.markdown(f"**Predictors:** {', '.join(bundle['config']['selected_preds'])}")
        st.markdown(f"**Distribution:** `{bundle['config']['dist']}`")
        st.markdown(f"**Variance Power:** `{bundle['config']['var_power']}`")
        exposure_col = bundle['config'].get('exposure_col', 'None')
        st.markdown(f"**Exposure Offset:** `{exposure_col}`")
        if exposure_col != 'None' and bundle['config'].get('use_target_base_rate', False):
            target_rate = bundle['config'].get('target_base_rate', 1.0)
            st.markdown(f"**Target Base Rate:** `{target_rate:.3f}`")
        st.markdown(f"**Timestamp:** `{bundle['config']['timestamp']}`")
        st.markdown(f"**Formula:** `{bundle['config']['formula']}`")

else:
    st.warning("No models saved yet. Fit a model and save it from this page.")

# --- Compare two saved models ---
if len(model_files) >= 2:
    st.subheader("📊 Compare Two Models")

    col1, col2 = st.columns(2)
    with col1:
        index1 = model_files.index(st.session_state["model1"]) if "model1" in st.session_state else 0
        model1 = st.selectbox("Model A", model_files, key="compare_model1", index=index1)
    with col2:
        index2 = model_files.index(st.session_state["model2"]) if "model2" in st.session_state else 0
        model2 = st.selectbox("Model B", model_files, key="compare_model2", index=index2)

    if model1 == model2:
        st.warning("Please select two different models to compare.")
    else:
        comparison_set = ["Model", "Holdout"]
        index3 = comparison_set.index(st.session_state["comparison_scope"]) if "comparison_scope" in st.session_state else 0
        comparison_scope = st.radio("Comparison set:", comparison_set, key="compare_scope", index=index3)

        def load_model_metadata(path):
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            return {
                "name": os.path.basename(path).replace(".pkl", ""),
                "dist": bundle["config"]["dist"],
                "var_power": bundle["config"]["var_power"],
                "response": bundle["config"]["response"],
                "predictors": bundle["config"]["selected_preds"],
                "exposure_col": bundle["config"].get("exposure_col", "None"),
                "use_target_base_rate": bundle["config"].get("use_target_base_rate", False),
                "target_base_rate": bundle["config"].get("target_base_rate", 1.0),
                "timestamp": bundle["config"]["timestamp"],
                "summary": bundle["summary"]
            }

        meta1 = load_model_metadata(f"saved_models/{model1}")
        meta2 = load_model_metadata(f"saved_models/{model2}")

        def extract_metrics(summary, subset):
            return {
                "📈 Fit Quality": {
                    "R-squared": summary.loc["R-squared", subset],
                    "Deviance Explained": summary.loc["Deviance Explained", subset]
                },
                "📉 Error Magnitude": {
                    "MAE": summary.loc["Mean Absolute Error", subset],
                    "MSE": summary.loc["Mean Squared Error", subset],
                    "RMSE": np.sqrt(summary.loc["Mean Squared Error", subset])
                },
                "🔍 Predictions": {
                    "Mean Actual": summary.loc["Mean Actual", subset],
                    "Mean Predicted": summary.loc["Mean Predicted", subset]
                },
                "🧮 Model Complexity": {
                    "n": summary.loc["n", subset]
                }
            }

        metrics1 = extract_metrics(meta1["summary"], comparison_scope)
        metrics2 = extract_metrics(meta2["summary"], comparison_scope)

        # Build grouped DataFrame
        rows = []
        for group in metrics1:
            rows.append((group, None, None))  # Section header
            for metric in metrics1[group]:
                rows.append((
                    metric,
                    metrics1[group][metric],
                    metrics2[group][metric]
                ))

        comparison_df = pd.DataFrame(
            rows,
            columns=["Metric", meta1["name"], meta2["name"]]
        ).set_index("Metric")

        st.markdown(f"### 🔍 Comparison on: **{comparison_scope} set**")
        st.dataframe(comparison_df.style.format("{:.4f}"), use_container_width=True)

        with st.expander("🧾 Metadata"):
            st.markdown(f"**{meta1['name']}**")
            st.markdown(f"- Response: `{meta1['response']}`")
            st.markdown(f"- Distribution: `{meta1['dist']}` (power={meta1['var_power']})")
            st.markdown(f"- Predictors: {', '.join(meta1['predictors'])}")
            st.markdown(f"- Exposure Offset: `{meta1['exposure_col']}`")
            if meta1['exposure_col'] != 'None' and meta1['use_target_base_rate']:
                st.markdown(f"- Target Base Rate: `{meta1['target_base_rate']:.3f}`")
            st.markdown("---")
            st.markdown(f"**{meta2['name']}**")
            st.markdown(f"- Response: `{meta2['response']}`")
            st.markdown(f"- Distribution: `{meta2['dist']}` (power={meta2['var_power']})")
            st.markdown(f"- Predictors: {', '.join(meta2['predictors'])}")
            st.markdown(f"- Exposure Offset: `{meta2['exposure_col']}`")
            if meta2['exposure_col'] != 'None' and meta2['use_target_base_rate']:
                st.markdown(f"- Target Base Rate: `{meta2['target_base_rate']:.3f}`")

        # Cache inputs
        st.session_state["model1"] = model1
        st.session_state["model2"] = model2
        st.session_state["comparison_scope"] = comparison_scope

# AI insights
get_ai_insights()

