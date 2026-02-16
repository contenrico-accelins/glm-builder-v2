import re
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.api import families
from statsmodels.genmod.families import Tweedie
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import BytesIO
from icecream import ic

# --- Helper Functions ---
def sanitize_name(name):
    """
    Convert arbitrary column names into valid Python identifiers:
    - Replace non-alphanumeric characters with '_'
    - Prefix a name starting with a digit with 'var_'
    """
    safe = re.sub(r"\W+", "_", name)
    if re.match(r"^\d", safe):
        safe = f"var_{safe}"
    return safe

# Ensure proper types
def clean_dataframe(df):
    # 1) Make a copy
    cleaned = df.copy()

    # 2) Standardize null-like placeholders
    null_like = ["", " ", "NA", "N/A", "NaN", "nan", "null", "NULL", "-", "--"]
    cleaned = cleaned.replace(to_replace=null_like, value=np.nan)
    cleaned = cleaned.infer_objects(copy=False)

    # 3) Identify columns to preserve as strings (e.g., names like "Grouped_Year")
    grouped_cols = [col for col in cleaned.columns if "grouped" in col.lower()]

    # 4) Coerce object columns to numeric where appropriate
    for col in cleaned.select_dtypes(include=["object"]).columns:
        if col not in grouped_cols:
            coerced = pd.to_numeric(cleaned[col], errors="coerce")
            if not coerced.isna().all():
                cleaned[col] = coerced

    # 5) Convert grouped columns to categorical with proper sorting
    for col in grouped_cols:
        # Skip if column is all NaN
        if cleaned[col].isna().all():
            continue
            
        # Get unique values, excluding NaN
        unique_vals = cleaned[col].dropna().unique()
        
        # Convert to string for consistent processing
        unique_vals_str = [str(x) for x in unique_vals]
        
        # Try to sort numerically if all values can be converted to numbers
        try:
            # Test if all values can be converted to float
            numeric_values = [float(x) for x in unique_vals_str]
            # If successful, sort by numeric value and convert back to string
            sorted_cats = [str(int(x)) if x.is_integer() else str(x) for x in sorted(numeric_values)]
            is_numeric_column = True
        except (ValueError, TypeError) as e:
            # If not all numeric, use alphabetical sorting
            sorted_cats = sorted(unique_vals_str)
            is_numeric_column = False
        
        # Convert the entire column to string first, but ensure consistency with categories
        string_column = cleaned[col].astype(str)
        
        # For numeric values, ensure they match the category format (integer if whole number)
        if col in grouped_cols and is_numeric_column:
            try:
                # Try to convert to numeric and back to ensure consistent string format
                numeric_series = pd.to_numeric(string_column, errors='coerce')
                string_column = numeric_series.apply(lambda x: str(int(x)) if pd.notna(x) and x == int(x) else str(x) if pd.notna(x) else 'nan')
            except Exception as e:
                pass  # Keep original string conversion if this fails
        
        cleaned[col] = string_column
        
        # Create categorical directly
        cleaned[col] = pd.Categorical(cleaned[col], categories=sorted_cats, ordered=True)

    return cleaned

def get_variable_type(series):
    """Determine if a variable should be treated as numeric or categorical"""
    # Drop NAs for dtype checking
    clean_series = series.dropna()
    
    # Check if it's already categorical
    if pd.api.types.is_categorical_dtype(clean_series):
        return "categorical"
    
    # Check if it's numeric
    if np.issubdtype(clean_series.dtype, np.number):
        return "numeric"
    
    # Default to categorical for everything else
    return "categorical"

def percent_usable(series):
    null_like = {"", " ", "NA", "N/A", "NaN", "nan", "null", "NULL"}
    # mask any entry in null_like as NaN, leave everything else untouched
    cleaned = series.where(~series.isin(null_like), other=np.nan)
    return round(100 * cleaned.notna().mean(), 1)

### --- GLM functions --- ###

def show_predictor_selector_sidebar():
    """
    Renders a predictor selector in the sidebar using a form.
    
    This function uses st.form to batch user selections from st.data_editor,
    preventing the app from re-rendering on every single checkbox click.
    State is only updated when the user explicitly clicks the submit button.
    """
    # Guard clause: Exit if the necessary data isn't in the session state yet.
    if "df_raw" not in st.session_state or "response_orig" not in st.session_state:
        return

    # Safely get the current predictor flags, defaulting to an empty list
    pred_flags = st.session_state.get("predictor_selector_state", [])
    
    # Determine all possible predictor columns
    all_preds = [
        c for c in st.session_state["df_raw"].columns 
        if c != st.session_state["response_orig"]
    ]

    # Build the dataframe for the editor. If state is out of sync, default to all False.
    if len(pred_flags) == len(all_preds):
        compact_df = pd.DataFrame({"variable": all_preds, "use_as_predictor": pred_flags})
    else:
        compact_df = pd.DataFrame({"variable": all_preds, "use_as_predictor": [False] * len(all_preds)})

    with st.sidebar:
        st.markdown("### 🧮 Predictor Selector")
        
        # Wrap the editor in a form
        with st.form(key="sidebar_predictor_form"):
            updated = st.data_editor(
                compact_df,
                column_config={
                    "use_as_predictor": st.column_config.CheckboxColumn(
                        "Predictor?", 
                        help="Select variables to use as predictors."
                    )
                },
                disabled=["variable"],
                hide_index=True,
                num_rows="fixed",
            )

            submitted = st.form_submit_button("Update Predictors")

        # Process the changes only when the form is submitted
        if submitted:
            # Get the list of selected predictor names
            selected_preds_list = updated.loc[updated["use_as_predictor"], "variable"].tolist()
            
            # Update all relevant session state variables at once
            st.session_state["selected_preds"] = selected_preds_list
            st.session_state["predictor_selector_state"] = updated["use_as_predictor"].tolist()
            st.session_state["preds"] = [st.session_state["col_map"][col] for col in selected_preds_list]
            
            # Trigger a re-run to ensure the rest of the app reflects the changes
            st.rerun()

def show_distribution():
    if "dist" not in st.session_state:
        return

    dist = st.session_state["dist"]
    var_power_ = 1.5  # default fallback
    if dist == "Tweedie":
        var_power_ = st.session_state.get("var_power", 1.5)

    with st.sidebar:
        st.markdown("### 📊 GLM Distribution")

        distributions = ["Gamma", "Gaussian", "Poisson", "Tweedie"]
        dist_ = st.selectbox(
            label="Select GLM distribution",
            options=distributions,
            index=distributions.index(dist)
        )

        if dist_ == "Tweedie":
            var_power = st.slider("Tweedie variance power", 1.0, 2.0, var_power_)
        else:
            var_power = None

    # Update session state with sidebar changes
    st.session_state["dist"] = dist_
    st.session_state["var_power"] = var_power


def optimize_tweedie(selected_preds, df_raw, df, response_orig):
    st.write("Evaluating different variance powers between 1.1 and 1.9...")

    # Rebuild formula
    terms = []
    for orig in selected_preds:
        col = sanitize_name(orig)
        if df_raw[orig].dtype.name in ['category', 'object']:
            terms.append(f"C({col})")
        else:
            terms.append(col)
    formula = f"{sanitize_name(response_orig)} ~ " + " + ".join(terms)

    # Evaluate models
    results = []
    for p in np.arange(1.1, 1.91, 0.1):
        try:
            model = smf.glm(
                formula=formula,
                data=df,
                family=Tweedie(var_power=p, link=families.links.log())
            ).fit()
            dev_explained = 1 - model.deviance / model.null_deviance
            results.append((round(p, 2), model.aic, dev_explained))
        except Exception:
            results.append((round(p, 2), np.nan, np.nan))

    df_results = pd.DataFrame(results, columns=["Variance Power", "AIC", "Deviance Explained"])
    df_results.set_index("Variance Power", inplace=True)

    # Conditional formatting: highlight best AIC and Deviance Explained
    def highlight_best(s, mode='min'):
        is_best = s == (s.min() if mode == 'min' else s.max())
        return ['background-color: #ffd700' if v else '' for v in is_best]

    styled = df_results.style \
        .format({"AIC": "{:.2f}", "Deviance Explained": "{:.3f}"}) \
        .apply(highlight_best, subset=["AIC"], mode='min') \
        .apply(highlight_best, subset=["Deviance Explained"], mode='max')

    st.dataframe(styled, column_config={
        "AIC": st.column_config.NumberColumn("AIC", help="AIC balances model fit and complexity — lower is better. Favor when you care about generalization."),
        "Deviance Explained": st.column_config.NumberColumn("Deviance Explained", help="Proportion of deviance explained — higher is better. Favor when you care most about in-sample performance")
        }
    )


def one_way_analysis(df, response, predictors, dist, var_power=None):
    results = []
    for pred in predictors:
        term = f"C({sanitize_name(pred)})" if df[pred].dtype.name in ['category', 'object'] else sanitize_name(pred)
        formula = f"{sanitize_name(response)} ~ {term}"
        if dist == 'Gamma':
            fam = sm.families.Gamma(link=sm.families.links.log())
        elif dist == 'Gaussian':
            fam = sm.families.Gaussian()
        elif dist == 'Poisson':
            fam = sm.families.Poisson()
        else:
            fam = Tweedie(var_power=var_power, link=sm.families.links.log())
        mod = smf.glm(formula=formula, data=df, family=fam).fit()
        p_val = mod.pvalues[1] if len(mod.pvalues) > 1 else mod.pvalues[0]
        results.append({
            'Predictor': pred,
            'p-value': p_val,
            'AIC': mod.aic,
            'BIC': mod.bic,
            'Deviance Explained': 1 - mod.deviance / mod.null_deviance
        })
    return pd.DataFrame(results).sort_values('Deviance Explained', ascending=False)


def plot_response_distribution(df, response):
    fig, ax = plt.subplots()
    sns.histplot(df[response], kde=True, ax=ax)
    ax.set_title(f"Distribution of {response}")
    st.pyplot(fig)


def plot_predictor_effect(df, response, predictor):
    fig, ax = plt.subplots()
    if df[predictor].dtype.name in ['category', 'object']:
        sns.barplot(x=predictor, y=response, data=df, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    else:
        sns.regplot(x=predictor, y=response, data=df, lowess=True, scatter_kws={'s':10}, ax=ax)
    ax.set_title(f"{response} vs {predictor}")
    st.pyplot(fig)

### --- GLM specific --- ###
def enforce_categories(dataframes: list, categorical_cols: list):
    """
    Ensures all specified categorical columns are consistent across a list of dataframes.
    Preserves custom order if the column is already an ordered categorical.
    """
    for col in categorical_cols:   
        all_levels = set()
        
        # Check if any dataframe has this column as an ordered categorical
        ordered_categories = None
        for df in dataframes:
            if col in df.columns and pd.api.types.is_categorical_dtype(df[col]) and df[col].cat.ordered:
                ordered_categories = df[col].cat.categories.tolist()
                break
        
        # Collect all unique values
        for df in dataframes:
            if col in df.columns:
                if pd.api.types.is_categorical_dtype(df[col]):
                    unique_vals = set(df[col].cat.categories.astype(str))
                else:
                    unique_vals = set(df[col].astype(str).unique())
                all_levels.update(unique_vals)

        # Determine final category order
        if ordered_categories:
            # Use the preserved order, adding any new categories at the end
            final_categories = [cat for cat in ordered_categories if cat in all_levels]
            for cat in sorted(all_levels):
                if cat not in final_categories:
                    final_categories.append(cat)
            final_categories.append("Other")
            final_categories = list(dict.fromkeys(final_categories))  # Remove duplicates while preserving order
            is_ordered = True
        else:
            # Use alphabetical order with "Other" at the end
            final_categories = sorted(list(all_levels)) + ["Other"]
            final_categories = sorted(list(set(final_categories)))
            is_ordered = False

        # Apply to all dataframes
        for df in dataframes:
            if col in df.columns:
                known_levels = all_levels
                df[col] = df[col].astype(str)
                df[col] = np.where(df[col].isin(known_levels), df[col], "Other")
                df[col] = pd.Categorical(df[col], categories=final_categories, ordered=is_ordered)

    return dataframes


def format_pricing_formula(params):
    # Check if an exposure offset is being used
    exposure_col = st.session_state.get("exposure_col", "None")
    use_exposure = exposure_col != "None"
    
    # Check if Target Base Rate is selected
    use_target_base_rate = st.session_state.get("use_target_base_rate", False)
    target_base_rate = st.session_state.get("target_base_rate", 1.0)
    
    base = np.exp(params.get("Intercept", 0))
    
    if use_exposure:
        # Display interpretable pricing formula with exposure offset
        st.markdown("#### 📋 **Pricing Formula Structure**")
        
        # Check if exposure column name suggests it's log-transformed
        is_log_exposure = "log" in exposure_col.lower()
        
        if is_log_exposure:
            # Extract the base variable name (e.g., "Buildings_SI" from "Log_Buildings_SI")
            base_var = exposure_col
            # Remove "log" prefix (case insensitive) and any underscores that follow
            base_var = re.sub(r'^log_?', '', base_var, flags=re.IGNORECASE)
            # Clean up any leading underscores
            base_var = base_var.lstrip('_')
            
            if use_target_base_rate:
                st.markdown(f"""
                **Premium = {base_var} × Intercept × Target Base Rate × Relativities**
                
                Where:
                - **{base_var}**: The exposure base (e.g., Sum Insured)
                - **Intercept**: {base:.4f}
                - **Target Base Rate**: {target_base_rate:.4f} (per unit of {base_var})
                - **Relativities**: Applied based on risk characteristics
                """)
            else:
                st.markdown(f"""
                **Premium = {base_var} × Base Rate × Relativities**
                
                Where:
                - **{base_var}**: The exposure base (e.g., Sum Insured)
                - **Base Rate**: {base:.4f} (per unit of {base_var})
                - **Relativities**: Applied based on risk characteristics
                """)
        else:
            st.markdown(f"""
            **Premium = exp({exposure_col} + Intercept + Risk Adjustments)**
            
            Where:
            - **{exposure_col}**: Exposure offset
            - **Base Factor**: {base:.4f}
            - **Risk Adjustments**: Sum of factor coefficients
            """)
    else:
        st.markdown(f"**Base premium**: `{base:.2f}`")

    # Build factor table
    df = pd.DataFrame({
        "Rating Factor": params.index,
        "Relativity": np.exp(params.values),
        "Coefficient": params.values
    })
    st.dataframe(df.style.format({
        "Relativity": "{:.3f}",
        "Coefficient": "{:.4f}"
    }), use_container_width=True)
    
    # Add example calculation if using exposure
    if use_exposure and len(params) > 1:  # Check if there are parameters beyond just intercept
        st.markdown("### 🧮 **Example Calculation**")
        
        # Create example with median/mode values
        if is_log_exposure:
            # Extract the base variable name (case insensitive removal of log prefix)
            base_var = re.sub(r'^log_?', '', exposure_col, flags=re.IGNORECASE).lstrip('_')
            
            if use_target_base_rate:
                st.markdown(f"""
                For a risk with **{base_var} = $100,000** and baseline risk characteristics:
                
                Premium = $100,000 × {base:.4f} × {target_base_rate:.4f} × (product of selected relativities)
                
                *Adjust relativities based on actual risk characteristics*
                """)
            else:
                st.markdown(f"""
                For a risk with **{base_var} = $100,000** and baseline risk characteristics:
                
                Premium = $100,000 × {base:.4f} × (product of selected relativities)
                
                *Adjust relativities based on actual risk characteristics*
                """)
        else:
            st.markdown(f"""
            Premium = exp({exposure_col} + {params.get("Intercept", 0):.4f} + sum of applicable coefficients)
            
            *Where coefficients are applied based on risk characteristics*
            """)

def summarize_fit(dfx, response, model):
    resid = dfx["residual"]
    y = dfx[response]
    yhat = dfx["predicted"]
    null_deviance = model.family.deviance(y, np.repeat(y.mean(), len(y)))
    model_deviance = model.family.deviance(y, yhat)
    deviance_explained = 1 - model_deviance / null_deviance if null_deviance > 0 else np.nan

    return {
        "n": len(dfx),
        "Mean Absolute Error": np.mean(np.abs(resid)),
        "Mean Squared Error": np.mean(resid ** 2),
        "R-squared": 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2),
        "Deviance": model_deviance,
        "Deviance Explained": deviance_explained,
        "Mean Actual": y.mean(),
        "Mean Predicted": yhat.mean()
    }

def residual_plot(dfx, label):
    fig, ax = plt.subplots()
    ax.scatter(dfx["predicted"], dfx["residual"], alpha=0.6)
    ax.axhline(0, linestyle="--", color="grey")
    ax.set_title(f"Residuals vs Fitted - {label}")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    return fig

def create_excel_report(writer):
    st.session_state["glm_df_full"].to_excel(writer, sheet_name="Full Data", index=False)
    st.session_state["glm_df_model"].to_excel(writer, sheet_name="Modelling Data", index=False)
    st.session_state["glm_df_holdout"].to_excel(writer, sheet_name="Holdout Data", index=False)
    st.session_state["glm_summary"].to_excel(writer, sheet_name="Summary")

    # Insert residual plots in Summary tab
    workbook  = writer.book
    worksheet = writer.sheets["Summary"]
    row_offset = st.session_state["glm_summary"].shape[0] + 3
    col_offset = 1
    for i, (label, fig) in enumerate(st.session_state["glm_plots"].items()):
        img_data = BytesIO()
        fig.savefig(img_data, format="png", dpi=150)
        img_data.seek(0)
        worksheet.insert_image(row_offset, col_offset + i * 10, f"{label}.png", {
            "image_data": img_data,
            "x_scale": 1,
            "y_scale": 1
        })

    # Pricing formula tab
    pricing_sheet = workbook.add_worksheet("Pricing Formula")
    base = np.exp(st.session_state["glm_model"].params.get("Intercept", 0))
    coefs = pd.DataFrame({
        "Factor": st.session_state["glm_model"].params.drop("Intercept", errors="ignore").index,
        "Log Coef": st.session_state["glm_model"].params.drop("Intercept", errors="ignore").values,
        "Relativity": np.exp(st.session_state["glm_model"].params.drop("Intercept", errors="ignore").values)
    })
    
    # Check if exposure offset is used
    exposure_col = st.session_state.get("exposure_col", "None")
    use_exposure = exposure_col != "None"
    is_log_exposure = use_exposure and "log" in exposure_col.lower()
    
    # Check if Target Base Rate is selected
    use_target_base_rate = st.session_state.get("use_target_base_rate", False)
    target_base_rate = st.session_state.get("target_base_rate", 1.0)
    
    # Enhanced pricing formula explanation
    row = 0
    if use_exposure and is_log_exposure:
        # Extract base variable name (case insensitive removal of log prefix)
        base_var = re.sub(r'^log_?', '', exposure_col, flags=re.IGNORECASE).lstrip('_')
        pricing_sheet.write(row, 0, "PRICING FORMULA STRUCTURE")
        
        if use_target_base_rate:
            pricing_sheet.write(row + 1, 0, f"Premium = {base_var} × Intercept × Target Base Rate × Product of Relativities")
        else:
            pricing_sheet.write(row + 1, 0, f"Premium = {base_var} × Base Rate × Product of Relativities")
            
        pricing_sheet.write(row + 2, 0, "(Example based on first row of dataset)")
        
        pricing_sheet.write(row + 4, 0, f"{base_var} (Exposure Base)")
        pricing_sheet.write(row + 4, 1, "Enter value here →")
        pricing_sheet.write(row + 4, 2, 100000)  # Example value
        
        # Simplified layout: put everything on the same row to avoid shifting issues
        if use_target_base_rate:
            pricing_sheet.write(row + 6, 0, "Base Rate (per unit)")
            pricing_sheet.write(row + 6, 1, base)
            pricing_sheet.write(row + 6, 3, "Intercept:")
            pricing_sheet.write(row + 6, 4, base)  # Same value as base rate
            pricing_sheet.write(row + 6, 6, "Target Base Rate:")
            pricing_sheet.write(row + 6, 7, target_base_rate)
        else:
            pricing_sheet.write(row + 6, 0, "Base Rate (per unit)")
            pricing_sheet.write(row + 6, 1, base)
        
        pricing_sheet.write(row + 8, 0, "RELATIVITIES BY FACTOR")
        pricing_sheet.write(row + 9, 0, "Factor")
        pricing_sheet.write(row + 9, 1, "Relativity")
        pricing_sheet.write(row + 9, 2, "Selected")
        pricing_sheet.write(row + 9, 3, "Applied Relativity")
        rel_start_row = row + 10
        
        # Get first row data for realistic example
        first_row = st.session_state["glm_df_full"].iloc[0] if len(st.session_state["glm_df_full"]) > 0 else None
        
        # Add relativities
        for i, (factor, rel, logcoef) in enumerate(zip(coefs["Factor"], coefs["Relativity"], coefs["Log Coef"])):
            factor_row = rel_start_row + i
            pricing_sheet.write(factor_row, 0, factor)
            
            # Relativities are unchanged - no adjustment needed
            pricing_sheet.write(factor_row, 1, rel)
            
            # Determine if this factor applies to first row
            selected = 0  # default to not selected
            if first_row is not None:
                # Check if this is a categorical factor (contains brackets)
                cat_pattern = r"C\((.*?)\)\[T\.(.*?)\]"
                match = re.match(cat_pattern, factor)
                
                if match:
                    # Categorical factor - check if first row matches this level
                    predictor_name = match.group(1)
                    factor_level = match.group(2)
                    if predictor_name in first_row.index:
                        # Convert both to string for comparison
                        first_row_value = str(first_row[predictor_name])
                        if first_row_value == factor_level:
                            selected = 1
                else:
                    # Numeric factor - always selected (continuous variables always apply)
                    selected = 1
            
            pricing_sheet.write(factor_row, 2, selected)
            pricing_sheet.write_formula(factor_row, 3, f"=IF(C{factor_row+1}=1,B{factor_row+1},1)")
        
        # Update Sum Insured to match first row if available
        if first_row is not None and exposure_col in first_row.index:
            # If exposure is log-transformed, we need to exp() it to get the actual value
            if is_log_exposure:
                actual_exposure = np.exp(first_row[exposure_col])
            else:
                actual_exposure = first_row[exposure_col]
            pricing_sheet.write(row + 4, 2, actual_exposure)
        
        # Final calculation
        final_row = rel_start_row + len(coefs) + 2
        pricing_sheet.write(final_row, 0, "FINAL PREMIUM CALCULATION")
        pricing_sheet.write(final_row + 1, 0, "Premium")
        
        # Formula: Sum Insured × [Base Rate or Intercept] × [Target Base Rate if applicable] × Product of Applied Relativities
        # The product range should cover all the "Applied Relativity" values in column D
        # Fix: Add 1 to account for 1-based Excel indexing
        product_range = f"D{rel_start_row + 1}:D{rel_start_row + len(coefs)}"
        if use_target_base_rate:
            # Premium = Sum Insured × Intercept × Target Base Rate × Product of Relativities
            # C{row+5} = Sum Insured, E{row+7} = Intercept, H{row+7} = Target Base Rate
            pricing_sheet.write_formula(final_row + 1, 1, f"=C{row+5}*E{row+7}*H{row+7}*PRODUCT({product_range})")
        else:
            # Premium = Sum Insured × Base Rate × Product of Relativities  
            # C{row+5} = Sum Insured, B{row+7} = Base Rate
            pricing_sheet.write_formula(final_row + 1, 1, f"=C{row+5}*B{row+7}*PRODUCT({product_range})")
        
        # Add explanation
        explanation_row = final_row + 3
        pricing_sheet.write(explanation_row, 0, "INSTRUCTIONS:")
        pricing_sheet.write(explanation_row + 1, 0, f"1. The example shows values from the first row of your dataset")
        pricing_sheet.write(explanation_row + 2, 0, f"2. Modify the {base_var} amount in cell C{row+5} as needed")
        pricing_sheet.write(explanation_row + 3, 0, "3. Set 'Selected' to 1 for applicable factors, 0 for not applicable")
        pricing_sheet.write(explanation_row + 4, 0, "4. The premium will calculate automatically")
        
    elif use_exposure:
        # Non-log exposure offset
        pricing_sheet.write(row, 0, "PRICING FORMULA STRUCTURE")
        pricing_sheet.write(row + 1, 0, f"Premium = exp({exposure_col} + Intercept + Sum of Applied Coefficients)")
        
        pricing_sheet.write(row + 3, 0, f"{exposure_col} (Exposure Offset)")
        pricing_sheet.write(row + 3, 1, "Enter value here →")
        pricing_sheet.write(row + 3, 2, 0)  # Default value
        
        pricing_sheet.write(row + 5, 0, "Intercept")
        pricing_sheet.write(row + 5, 1, st.session_state["glm_model"].params.get("Intercept", 0))
        
        pricing_sheet.write(row + 7, 0, "COEFFICIENTS BY FACTOR")
        pricing_sheet.write(row + 8, 0, "Factor")
        pricing_sheet.write(row + 8, 1, "Coefficient")
        pricing_sheet.write(row + 8, 2, "Selected")
        pricing_sheet.write(row + 8, 3, "Applied Coefficient")
        
        # Get first row data for realistic example
        first_row = st.session_state["glm_df_full"].iloc[0] if len(st.session_state["glm_df_full"]) > 0 else None
        
        # Add coefficients
        for i, (factor, rel, logcoef) in enumerate(zip(coefs["Factor"], coefs["Relativity"], coefs["Log Coef"])):
            factor_row = row + 9 + i
            pricing_sheet.write(factor_row, 0, factor)
            pricing_sheet.write(factor_row, 1, logcoef)
            
            # Determine if this factor applies to first row
            selected = 0  # default to not selected
            if first_row is not None:
                # Check if this is a categorical factor (contains brackets)
                cat_pattern = r"C\((.*?)\)\[T\.(.*?)\]"
                match = re.match(cat_pattern, factor)
                
                if match:
                    # Categorical factor - check if first row matches this level
                    predictor_name = match.group(1)
                    factor_level = match.group(2)
                    if predictor_name in first_row.index:
                        # Convert both to string for comparison
                        first_row_value = str(first_row[predictor_name])
                        if first_row_value == factor_level:
                            selected = 1
                else:
                    # Numeric factor - always selected (continuous variables always apply)
                    selected = 1
            
            pricing_sheet.write(factor_row, 2, selected)
            pricing_sheet.write_formula(factor_row, 3, f"=IF(C{factor_row+1}=1,B{factor_row+1},0)")
        
        # Update exposure value to match first row if available
        if first_row is not None and exposure_col in first_row.index:
            pricing_sheet.write(row + 3, 2, first_row[exposure_col])
        
        # Final calculation
        final_row = row + 9 + len(coefs) + 2
        pricing_sheet.write(final_row, 0, "FINAL PREMIUM CALCULATION")
        pricing_sheet.write(final_row + 1, 0, "Premium")
        # Formula: exp(exposure + intercept + sum of applied coefficients)
        sum_range = f"D{row+9+1}:D{row+9+len(coefs)}"
        pricing_sheet.write_formula(final_row + 1, 1, f"=EXP(C{row+4}+B{row+6}+SUM({sum_range}))")
        
    else:
        # No exposure offset - traditional approach
        pricing_sheet.write(row, 0, "Base Premium")
        pricing_sheet.write(row, 1, base)
        pricing_sheet.write(row + 2, 0, "Factor")
        pricing_sheet.write(row + 2, 1, "Relativity")
        pricing_sheet.write(row + 2, 2, "Selected")
        pricing_sheet.write(row + 2, 3, "Applied Log Coef")

        # Get first row data for realistic example
        first_row = st.session_state["glm_df_full"].iloc[0] if len(st.session_state["glm_df_full"]) > 0 else None

        for i, (factor, rel, logcoef) in enumerate(zip(coefs["Factor"], coefs["Relativity"], coefs["Log Coef"])):
            factor_row = row + 3 + i
            pricing_sheet.write(factor_row, 0, factor)
            pricing_sheet.write(factor_row, 1, rel)
            
            # Determine if this factor applies to first row
            selected = 0  # default to not selected
            if first_row is not None:
                # Check if this is a categorical factor (contains brackets)
                cat_pattern = r"C\((.*?)\)\[T\.(.*?)\]"
                match = re.match(cat_pattern, factor)
                
                if match:
                    # Categorical factor - check if first row matches this level
                    predictor_name = match.group(1)
                    factor_level = match.group(2)
                    if predictor_name in first_row.index:
                        # Convert both to string for comparison
                        first_row_value = str(first_row[predictor_name])
                        if first_row_value == factor_level:
                            selected = 1
                else:
                    # Numeric factor - always selected (continuous variables always apply)
                    selected = 1
            
            pricing_sheet.write(factor_row, 2, selected)
            pricing_sheet.write_formula(factor_row, 3, f"=LN(B{factor_row+1})*C{factor_row+1}")

        pricing_sheet.write(len(coefs) + 4, 0, "Final Premium:")
        pricing_sheet.write_formula(
            len(coefs) + 4, 1, f"=B1*EXP(SUM(D4:D{len(coefs)+3}))"
        )


## Cache inputs
def check_if_model_inputs_changed(current_inputs: dict, cache_key="glm_last_fit_config") -> bool:
    prev_inputs = st.session_state.get(cache_key, {})
    st.session_state["model_inputs_changed"] = current_inputs != prev_inputs
    return st.session_state["model_inputs_changed"]

def save_model_input_snapshot(current_inputs: dict, cache_key="glm_last_fit_config"):
    st.session_state[cache_key] = current_inputs


### --- Model Manager --- ###
def save_glm_to_disk(name="model_1"):
    model_bundle = {
        "df_raw": st.session_state["df_raw"],
        "df": st.session_state["df"],
        "col_map": st.session_state["col_map"],
        "model": st.session_state["glm_model"],
        "df_model": st.session_state["glm_df_model"],
        "df_holdout": st.session_state["glm_df_holdout"],
        "df_full": st.session_state["glm_df_full"],
        "summary": st.session_state["glm_summary"],
        "plots": st.session_state["glm_plots"],
        "config": {
            "dist": st.session_state["dist"],
            "var_power": st.session_state["var_power"],
            "response": st.session_state["response_orig"],
            "response_sanitized": st.session_state["response"],
            "selected_preds": st.session_state["selected_preds"],
            "sanitized_preds": st.session_state["preds"],
            "holdout_frac": st.session_state["holdout_frac"],
            "exposure_col": st.session_state["exposure_col"],
            "use_target_base_rate": st.session_state.get("use_target_base_rate", False),
            "target_base_rate": st.session_state.get("target_base_rate", 1.0),
            "predictor_selector_state": st.session_state.get("predictor_selector_state", []),
            "formula": str(st.session_state["glm_model"].model.formula),
            "timestamp": datetime.now().isoformat(),
            "input_snapshot": st.session_state.get("glm_last_fit_config", {})
        },
        # Save one-way analysis and category ordering
        "one_way_analysis": {
            "summary": st.session_state.get("one_way_summary"),
            "df_reordered": st.session_state.get("df_reordered"),
            "input_snapshot": st.session_state.get("glm_last_fit_config", {})  # Use same snapshot for consistency
        }
    }

    Path("saved_models").mkdir(exist_ok=True)
    with open(f"saved_models/{name}.pkl", "wb") as f:
        pickle.dump(model_bundle, f)

def load_glm_from_disk(name="model_1"):
    with open(f"saved_models/{name}.pkl", "rb") as f:
        model_bundle = pickle.load(f)

    config = model_bundle["config"]

    # Restore core data + config
    st.session_state["df_raw"] = model_bundle["df_raw"]
    st.session_state["df"] = model_bundle["df"]
    st.session_state["col_map"] = model_bundle["col_map"]

    st.session_state["glm_model"] = model_bundle["model"]
    st.session_state["glm_df_model"] = model_bundle["df_model"]
    st.session_state["glm_df_holdout"] = model_bundle["df_holdout"]
    st.session_state["glm_df_full"] = model_bundle["df_full"]
    st.session_state["glm_summary"] = model_bundle["summary"]
    st.session_state["glm_plots"] = model_bundle["plots"]

    # Restore config details
    st.session_state["dist"] = config["dist"]
    st.session_state["var_power"] = config["var_power"]
    st.session_state["response_orig"] = config["response"]
    st.session_state["response"] = config["response_sanitized"]
    st.session_state["selected_preds"] = config["selected_preds"]
    st.session_state["preds"] = config["sanitized_preds"]
    st.session_state["holdout_frac"] = config["holdout_frac"]
    st.session_state["exposure_col"] = config["exposure_col"]
    st.session_state["use_target_base_rate"] = config.get("use_target_base_rate", False)
    st.session_state["target_base_rate"] = config.get("target_base_rate", 1.0)
    st.session_state["predictor_selector_state"] = config.get("predictor_selector_state", [False] * len(config["selected_preds"]))
    st.session_state["glm_last_fit_config"] = config.get("input_snapshot", {})
    
    # Restore one-way analysis and category ordering if available
    one_way_data = model_bundle.get("one_way_analysis", {})
    if one_way_data.get("summary") is not None:
        st.session_state["one_way_summary"] = one_way_data["summary"]
    if one_way_data.get("df_reordered") is not None:
        st.session_state["df_reordered"] = one_way_data["df_reordered"]

### --- Pricing Comparison --- ###

def extract_relativities(params1, params2, name1="Model A", name2="Model B"):
    """
    Compares the relativities of two GLM models by parsing their coefficients.
    The Intercept (Base Premium) is ignored.
    
    Args:
        params1 (pd.Series): Coefficients from the first model.
        params2 (pd.Series): Coefficients from the second model.
        name1 (str): Name for the first model.
        name2 (str): Name for the second model.
        
    Returns:
        pd.DataFrame: A dataframe comparing the relativities.
    """
    
    def process_params(params):
        """Helper to parse a single model's parameters into a structured DataFrame."""
        records = []
        cat_pattern = re.compile(r"C\((.*?)\)\[T\.(.*?)\]")
        
        for idx, value in params.items():
            # --- MODIFICATION: Skip the intercept ---
            if idx == "Intercept":
                continue

            relativity = np.exp(value)
            factor_type = "Continuous"
            
            match = cat_pattern.match(idx)
            if match:
                factor, level = match.groups()
                factor_type = "Categorical"
            else: # Continuous variable
                factor, level = idx, "per unit"

            records.append({
                "Factor": factor,
                "Level": level,
                "Relativity": relativity,
                "Factor Type": factor_type,
            })
            
        return pd.DataFrame(records)

    df1 = process_params(params1)
    df2 = process_params(params2)
    
    # Merge the two dataframes on all identifying columns
    comparison_df = pd.merge(
        df1, df2, 
        on=["Factor", "Level", "Factor Type"], 
        how="outer", 
        suffixes=(f"_{name1}", f"_{name2}")
    )
    
    # Rename columns and fill missing values
    comparison_df.rename(columns={
        f"Relativity_{name1}": name1,
        f"Relativity_{name2}": name2,
    }, inplace=True)
    
    comparison_df[name1].fillna(1.0, inplace=True)
    comparison_df[name2].fillna(1.0, inplace=True)
    
    comparison_df['Ratio (B/A)'] = comparison_df[name2] / comparison_df[name1]
    
    # Sort and reorder for clarity
    comparison_df = comparison_df.sort_values(by=["Factor Type", "Factor", "Level"]).reset_index(drop=True)
    final_cols = ['Factor', 'Level', 'Factor Type', name1, name2, 'Ratio (B/A)']
    
    return comparison_df[final_cols]


def plot_relativity_comparison(plot_data, factor_name, name1="Model A", name2="Model B"):
    """
    Generates a grouped bar chart comparing relativities for a single categorical factor.
    """
    
    levels = plot_data.index
    x = np.arange(len(levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, plot_data[name1], width, label=name1, color='cornflowerblue')
    rects2 = ax.bar(x + width/2, plot_data[name2], width, label=name2, color='sandybrown')

    ax.set_ylabel('Relativity (Factor Loading)')
    ax.set_title(f'Relativity Comparison for Factor: {factor_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=45, ha="right")
    ax.legend()
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)

    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=9)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=9)

    fig.tight_layout()
    
    return fig


def plot_numeric_factor_comparison(plot_data, name1="Model A", name2="Model B"):
    """
    --- NEW FUNCTION ---
    Generates a grouped bar chart comparing relativities for all numeric factors.
    """
    plot_data = plot_data.set_index('Factor')
    factors = plot_data.index
    x = np.arange(len(factors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width/2, plot_data[name1], width, label=name1, color='cornflowerblue')
    rects2 = ax.bar(x + width/2, plot_data[name2], width, label=name2, color='sandybrown')

    ax.set_ylabel('Relativity (per unit change)')
    ax.set_title('Relativity Comparison for Numeric Factors')
    ax.set_xticks(x)
    ax.set_xticklabels(factors, rotation=45, ha="right")
    ax.legend()
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)

    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=9)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=9)
    
    fig.tight_layout()
    
    return fig


def unify_and_prepare_dataset(df_to_prepare, model1, model2):
    """
    Prepares a dataframe for scoring against two potentially different models.
    It identifies all required predictors, adds missing categorical columns,
    and unifies the levels of shared categorical columns.

    Args:
        df_to_prepare (pd.DataFrame): The raw dataframe to be scored.
        model1 (statsmodels.GLMResultsWrapper): The first fitted model object.
        model2 (statsmodels.GLMResultsWrapper): The second fitted model object.

    Returns:
        pd.DataFrame: A cleaned and prepared dataframe ready for prediction.
        
    Raises:
        ValueError: If a required numeric predictor is missing from the dataframe.
    """
    prepared_df = df_to_prepare.copy()
    
    # 1. CORRECTED: Get all unique required predictors by parsing exog_names
    all_preds = {} # Use a dict to store name and type (numeric/categorical)
    cat_pattern = re.compile(r"C\((.*?)\)\[T\..*?\]")

    for model in [model1, model2]:
        for exog_name in model.model.exog_names:
            if exog_name == 'Intercept':
                continue
            
            match = cat_pattern.match(exog_name)
            if match:
                # Categorical variable found, e.g., from "C(Roof_Type)[T.Slate]"
                factor_name = match.group(1)
                all_preds[factor_name] = 'categorical'
            else:
                # Numeric variable found, e.g., "Log_Buildings_SI"
                factor_name = exog_name
                all_preds[factor_name] = 'numeric'
    
    # 2. Check for missing columns and unify categories
    for pred, pred_type in all_preds.items():
        if pred not in prepared_df.columns:
            if pred_type == 'numeric':
                # We can't safely invent a numeric column.
                raise ValueError(f"The selected dataset is missing a required numeric predictor: '{pred}'")
            else: # Add missing categorical column and fill with "Other"
                prepared_df[pred] = "Other"

    # 3. For all categorical predictors, unify their levels
    categorical_preds = [p for p, t in all_preds.items() if t == 'categorical']
    
    for cat_pred in categorical_preds:
        # Gather all known levels from both models for this predictor
        union_of_levels = set()
        for model in [model1, model2]:
            # Find the factor info, which is keyed by the predictor name
            if cat_pred in model.model.data.design_info.factor_infos:
                union_of_levels.update(model.model.data.design_info.factor_infos[cat_pred].categories)
        
        # Add "Other" to handle all unforeseen cases
        union_of_levels.add("Other")
        final_categories = sorted(list(union_of_levels))
        
        # Ensure the column is of string type before checking levels
        prepared_df[cat_pred] = prepared_df[cat_pred].astype(str)
        
        # Map any level not in our master list to "Other"
        known_levels = set(final_categories)
        prepared_df[cat_pred] = prepared_df[cat_pred].apply(lambda x: x if x in known_levels else "Other")

        # Set the column to the unified categorical type
        prepared_df[cat_pred] = pd.Categorical(
            prepared_df[cat_pred],
            categories=final_categories,
            ordered=False
        )
        
    return prepared_df

# --- Category Reordering Functions ---

def extract_predictor_categories(df, predictors):
    """
    Extract all categories for categorical predictors in the dataframe.
    Returns a DataFrame with predictor names and their categories with current order.
    """
    category_data = []
    
    for pred in predictors:
        if pred in df.columns:
            if df[pred].dtype.name in ['category', 'object']:
                # Get unique categories
                if pd.api.types.is_categorical_dtype(df[pred]):
                    categories = df[pred].cat.categories.tolist()
                else:
                    categories = sorted(df[pred].dropna().unique().tolist())
                
                # Create rows for each category with its current order
                for i, category in enumerate(categories):
                    category_data.append({
                        'Predictor': pred,
                        'Category': str(category),
                        'Order': i + 1
                    })
    
    return pd.DataFrame(category_data)

def apply_category_reordering(df, category_df):
    """
    Apply category reordering to dataframe based on the uploaded category DataFrame.
    Uses CategoricalDtype to ensure proper ordering for GLM fitting.
    Returns a modified copy of the dataframe.
    """
    from pandas.api.types import CategoricalDtype
    
    df_modified = df.copy()
    
    # Group by predictor
    for predictor in category_df['Predictor'].unique():
        if predictor in df_modified.columns:
            pred_categories = category_df[category_df['Predictor'] == predictor]
            
            # Sort by order and get the category list
            ordered_categories = pred_categories.sort_values('Order')['Category'].tolist()
            
            # Get current categories in the data
            if df_modified[predictor].dtype.name in ['category', 'object']:
                if pd.api.types.is_categorical_dtype(df_modified[predictor]):
                    current_categories = set(df_modified[predictor].cat.categories.astype(str))
                else:
                    current_categories = set(df_modified[predictor].dropna().astype(str).unique())
                
                # Check if the categories in the CSV actually exist in the data
                csv_categories = set(ordered_categories)
                missing_categories = csv_categories - current_categories
                
                if missing_categories:
                    st.warning(f"⚠️ Categories {missing_categories} specified in CSV for '{predictor}' don't exist in the data. Current categories: {sorted(current_categories)}")
                    # Filter out missing categories
                    ordered_categories = [cat for cat in ordered_categories if cat in current_categories]
                
                if ordered_categories:
                    # Create CategoricalDtype with the desired order
                    cat_type = CategoricalDtype(categories=ordered_categories, ordered=True)
                    # Convert the column to the new categorical type
                    df_modified[predictor] = df_modified[predictor].astype(cat_type)
    
    return df_modified

def create_category_csv_download(df, predictors):
    """
    Create a CSV download button for predictor categories.
    """
    category_df = extract_predictor_categories(df, predictors)
    
    if not category_df.empty:
        csv_buffer = BytesIO()
        category_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Category Order CSV",
            data=csv_data,
            file_name=f"predictor_categories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download current category order for all categorical predictors"
        )
        
        return category_df
    else:
        st.info("No categorical predictors found.")
        return pd.DataFrame()

def upload_category_csv_and_preview(df, predictors):
    """
    Handle CSV upload for category reordering and show preview.
    Returns the uploaded category DataFrame if valid, otherwise None.
    """
    uploaded_file = st.file_uploader(
        "📤 Upload Category Order CSV", 
        type=['csv'],
        help="Upload a CSV with columns: Predictor, Category, Order"
    )
    
    if uploaded_file is not None:
        try:
            category_df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['Predictor', 'Category', 'Order']
            if not all(col in category_df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                return None
            
            # Validate that predictors exist in the dataframe
            invalid_predictors = set(category_df['Predictor'].unique()) - set(predictors)
            if invalid_predictors:
                st.warning(f"Unknown predictors in CSV will be ignored: {', '.join(invalid_predictors)}")
                # Filter out invalid predictors
                category_df = category_df[category_df['Predictor'].isin(predictors)]
            
            if category_df.empty:
                st.error("No valid predictors found in the uploaded CSV.")
                return None
            
            return category_df
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            return None
    
    return None

def show_category_reorder_preview(df, category_df, predictors):
    """
    Show a preview of what the category reordering will look like.
    """
    st.subheader("🔍 Preview of Category Reordering")
    
    # Show comparison table
    comparison_data = []
    
    for predictor in category_df['Predictor'].unique():
        if predictor in predictors and predictor in df.columns:
            if df[predictor].dtype.name in ['category', 'object']:
                # Current order
                if pd.api.types.is_categorical_dtype(df[predictor]):
                    current_order = df[predictor].cat.categories.tolist()
                else:
                    current_order = sorted(df[predictor].dropna().unique().tolist())
                
                # New order from CSV
                pred_categories = category_df[category_df['Predictor'] == predictor]
                new_order = pred_categories.sort_values('Order')['Category'].tolist()
                
                comparison_data.append({
                    'Predictor': predictor,
                    'Current Order': ' → '.join(current_order),
                    'New Order': ' → '.join(new_order)
                })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(
            comparison_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Predictor": st.column_config.Column(
                    "Predictor",
                    pinned=True
                )
            }
        )
        
        return True
    else:
        st.warning("No categorical predictors found to reorder.")
        return False

def apply_category_reordering_confirmed(df, category_df):
    """
    Apply category reordering after confirmation.
    Returns modified dataframe.
    """
    df_reordered = apply_category_reordering(df, category_df)
    
    return df_reordered

# AI insights
def get_ai_insights():
    with st.sidebar:
        st.markdown("### ✨ AI Insights")
        if st.button("Get AI Insights"):
            pass

def parse_factor_name(factor_name):
    """
    Parse a factor name from uploaded relativities CSV to extract column and level.
    
    Args:
        factor_name (str): Factor name like "C(Area_grouped)[T.2]" or "Log_Sum_Insured"
        
    Returns:
        tuple: (column_name, level, factor_type)
               - For categorical: ("Area_grouped", "2", "Categorical")
               - For continuous: ("Log_Sum_Insured", "per unit", "Continuous")
    """
    cat_pattern = re.compile(r"C\((.*?)\)\[T\.(.*?)\]")
    match = cat_pattern.match(factor_name)
    
    if match:
        column_name = match.group(1)
        level = match.group(2)
        return column_name, level, "Categorical"
    else:
        # Continuous factor
        return factor_name, "per unit", "Continuous"

def apply_relativities_to_dataset(dataset, relativities_df, base_rate, exposure_col):
    """
    Apply relativities to calculate premiums for Model A and B from uploaded CSV format.
    Uses the formula: Premium = Exposure × Base Rate × Relativities
    
    Args:
        dataset (pd.DataFrame): The scoring dataset
        relativities_df (pd.DataFrame): CSV with 3 columns: [Factor, Model_A_Values, Model_B_Values]
        base_rate (float): Base rate per unit of exposure
        exposure_col (str): Column name for exposure (e.g., Sum_Insured)
        
    Returns:
        pd.DataFrame: Dataset with Premium_A and Premium_B columns added
    """
    # Initialize with base premium calculation: Exposure × Base Rate
    dataset_copy = dataset.copy()
    
    # Check if exposure column exists
    if exposure_col in dataset_copy.columns:
        try:
            # Premium = Exposure × Base Rate × Relativities (starting with relativities = 1.0)
            dataset_copy['Premium_A'] = dataset_copy[exposure_col] * base_rate
            dataset_copy['Premium_B'] = dataset_copy[exposure_col] * base_rate
            st.info(f"✅ Using exposure column '{exposure_col}' with base rate {base_rate:.4f} per unit")
        except Exception as e:
            st.error(f"❌ Error calculating base premium: {str(e)}")
            return dataset_copy
    else:
        st.error(f"❌ Exposure column '{exposure_col}' not found in dataset. Available columns: {list(dataset_copy.columns)}")
        return dataset_copy
    
    # Get column names (should be standardized to Factor, Model_A_Values, Model_B_Values)
    factor_col = relativities_df.columns[0]
    model_a_col = relativities_df.columns[1] 
    model_b_col = relativities_df.columns[2]
    
    # Apply each relativity multiplicatively
    for _, row in relativities_df.iterrows():
        factor_name = row[factor_col]
        model_a_rel = row[model_a_col]
        model_b_rel = row[model_b_col]
        
        # Parse factor name to get column and level
        column_name, level, factor_type = parse_factor_name(factor_name)
        
        if factor_type == "Categorical":
            if column_name in dataset_copy.columns:
                # Apply relativity to matching records (multiply by the relativity)
                mask = dataset_copy[column_name].astype(str) == level
                dataset_copy.loc[mask, 'Premium_A'] *= model_a_rel
                dataset_copy.loc[mask, 'Premium_B'] *= model_b_rel
            else:
                st.warning(f"⚠️ Column '{column_name}' not found in dataset. Skipping factor '{factor_name}'.")
        
        elif factor_type == "Continuous":
            # For continuous factors, we would need the actual data values and apply the coefficient
            # This is more complex and would require the coefficient, not just the relativity
            st.info(f"ℹ️ Skipping continuous factor '{factor_name}' - continuous factors require coefficient application which is not supported in this mode.")
            continue
    
    # Show summary of calculation
    st.success(f"✅ Applied {len(relativities_df)} relativities to {len(dataset_copy)} records")
    
    return dataset_copy


def apply_relativities_to_dataset_for_audit_upload(dataset, relativities_df, model_a_name, model_b_name):
    """
    Apply relativities to dataset and return individual factor relativities for audit (Upload mode).
    
    Args:
        dataset (pd.DataFrame): The scoring dataset
        relativities_df (pd.DataFrame): CSV with 3 columns: [Factor, Model_A_Values, Model_B_Values]
        model_a_name (str): Name for Model A columns
        model_b_name (str): Name for Model B columns
        
    Returns:
        pd.DataFrame: Dataset with individual relativity columns for each factor
    """
    dataset_copy = dataset.copy()
    
    # Get column names
    factor_col = relativities_df.columns[0]
    model_a_col = relativities_df.columns[1] 
    model_b_col = relativities_df.columns[2]
    
    # Get unique factors (not factor-level combinations)
    unique_factors = set()
    for _, row in relativities_df.iterrows():
        factor_name = row[factor_col]
        column_name, level, factor_type = parse_factor_name(factor_name)
        if factor_type == "Categorical":
            unique_factors.add(column_name)
    
    factors_processed = []
    factors_not_found = []
    
    # Create one column per factor
    for factor_name in unique_factors:
        # Check if factor exists in dataset
        factor_in_dataset = None
        if factor_name in dataset_copy.columns:
            factor_in_dataset = factor_name
        else:
            # Try to find a sanitized version or similar name
            sanitized_factor = sanitize_name(factor_name)
            if sanitized_factor in dataset_copy.columns:
                factor_in_dataset = sanitized_factor
            else:
                # Try reverse - check if any dataset column sanitizes to this factor
                for col in dataset_copy.columns:
                    if sanitize_name(col) == factor_name:
                        factor_in_dataset = col
                        break
        
        if factor_in_dataset:
            factors_processed.append(factor_name)
            
            # Updated naming convention
            model_a_rel_col = f"Rel_{factor_name}_A"
            model_b_rel_col = f"Rel_{factor_name}_B"
            
            # Initialize with base relativity (1.0)
            dataset_copy[model_a_rel_col] = 1.0
            dataset_copy[model_b_rel_col] = 1.0
            
            # Apply relativities for each level of this factor
            for _, row in relativities_df.iterrows():
                factor_name_full = row[factor_col]
                column_name, level, factor_type = parse_factor_name(factor_name_full)
                
                if column_name == factor_name and factor_type == "Categorical":
                    model_a_rel = row[model_a_col]
                    model_b_rel = row[model_b_col]
                    
                    # Apply to matching records
                    mask = dataset_copy[factor_in_dataset].astype(str) == level
                    dataset_copy.loc[mask, model_a_rel_col] = model_a_rel
                    dataset_copy.loc[mask, model_b_rel_col] = model_b_rel
        else:
            factors_not_found.append(factor_name)
    
    # Simple confirmation message
    final_rel_cols = [col for col in dataset_copy.columns if col.startswith('Rel_')]
    if final_rel_cols:
        st.success(f"✅ Added relativities for {len(factors_processed)} factors ({len(final_rel_cols)} columns)")
    
    return dataset_copy


def apply_relativities_to_dataset_for_audit(dataset, comparison_df, model1, model2, model_a_name, model_b_name):
    """
    Apply relativities to dataset and return individual factor relativities for audit (Saved Models mode).
    
    Args:
        dataset (pd.DataFrame): The scoring dataset (with original column names)
        comparison_df (pd.DataFrame): Comparison dataframe with relativities
        model1: First GLM model
        model2: Second GLM model
        model_a_name (str): Name for Model A columns
        model_b_name (str): Name for Model B columns
        
    Returns:
        pd.DataFrame: Dataset with individual relativity columns for each factor
    """
    dataset_copy = dataset.copy()
    
    # Get all categorical factors
    categorical_factors = comparison_df[comparison_df['Factor Type'] == 'Categorical']
    
    factors_processed = []
    factors_not_found = []
    
    # Process each unique factor (not factor-level combinations)
    for factor_name in categorical_factors['Factor'].unique():
        # Check if factor exists in dataset (exact match or with sanitization)
        factor_in_dataset = None
        if factor_name in dataset_copy.columns:
            factor_in_dataset = factor_name
        else:
            # Try to find a sanitized version or similar name
            sanitized_factor = sanitize_name(factor_name)
            if sanitized_factor in dataset_copy.columns:
                factor_in_dataset = sanitized_factor
            else:
                # Try reverse - check if any dataset column sanitizes to this factor
                for col in dataset_copy.columns:
                    if sanitize_name(col) == factor_name:
                        factor_in_dataset = col
                        break
        
        if factor_in_dataset:
            factors_processed.append(factor_name)
            factor_data = categorical_factors[categorical_factors['Factor'] == factor_name]
            
            # Create one column per factor per model - Updated naming convention
            model_a_rel_col = f"Rel_{factor_name}_A"
            model_b_rel_col = f"Rel_{factor_name}_B"
            
            # Initialize with base relativity (1.0) for all records
            dataset_copy[model_a_rel_col] = 1.0
            dataset_copy[model_b_rel_col] = 1.0
            
            # Apply relativities for each level
            for _, row in factor_data.iterrows():
                level = row['Level']
                model_a_rel = row[model_a_name]
                model_b_rel = row[model_b_name]
                
                # Apply to matching records
                mask = dataset_copy[factor_in_dataset].astype(str) == level
                dataset_copy.loc[mask, model_a_rel_col] = model_a_rel
                dataset_copy.loc[mask, model_b_rel_col] = model_b_rel
        else:
            factors_not_found.append(factor_name)
    
    # Handle continuous factors (show coefficient application)
    continuous_factors = comparison_df[comparison_df['Factor Type'] == 'Continuous']
    for _, row in continuous_factors.iterrows():
        factor_name = row['Factor']
        
        # Check if factor exists in dataset
        factor_in_dataset = None
        if factor_name in dataset_copy.columns:
            factor_in_dataset = factor_name
        else:
            # Try to find a sanitized version or similar name
            sanitized_factor = sanitize_name(factor_name)
            if sanitized_factor in dataset_copy.columns:
                factor_in_dataset = sanitized_factor
            else:
                # Try reverse - check if any dataset column sanitizes to this factor
                for col in dataset_copy.columns:
                    if sanitize_name(col) == factor_name:
                        factor_in_dataset = col
                        break
        
        if factor_in_dataset:
            factors_processed.append(f"{factor_name} (continuous)")
            model_a_coef = row[model_a_name]
            model_b_coef = row[model_b_name]
            
            # Updated naming convention
            model_a_rel_col = f"Rel_{factor_name}_A"
            model_b_rel_col = f"Rel_{factor_name}_B"
            
            # For continuous factors, relativity = exp(coefficient * value)
            dataset_copy[model_a_rel_col] = np.exp(model_a_coef * dataset_copy[factor_in_dataset])
            dataset_copy[model_b_rel_col] = np.exp(model_b_coef * dataset_copy[factor_in_dataset])
        else:
            factors_not_found.append(f"{factor_name} (continuous)")
    
    # Simple confirmation message
    final_rel_cols = [col for col in dataset_copy.columns if col.startswith('Rel_')]
    if final_rel_cols:
        st.success(f"✅ Added relativities for {len(factors_processed)} factors ({len(final_rel_cols)} columns)")
    
    return dataset_copy