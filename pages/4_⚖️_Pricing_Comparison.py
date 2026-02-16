import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from utils import (
    extract_relativities, 
    plot_relativity_comparison, 
    plot_numeric_factor_comparison, 
    unify_and_prepare_dataset, 
    get_ai_insights,
    apply_relativities_to_dataset,
    parse_factor_name,
    sanitize_name,
    apply_relativities_to_dataset_for_audit,
    apply_relativities_to_dataset_for_audit_upload
)

st.set_page_config(page_title="GLM Builder - Pricing Comparison", layout="centered", page_icon="⚖️")
st.title("⚖️ Pricing Comparison")

# Initialize session state variables
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = "Compare Saved Models"
if 'model_names' not in st.session_state:
    st.session_state.model_names = {}
if 'pred_data' not in st.session_state:
    st.session_state.pred_data = {}
if 'comparison_df' not in st.session_state:
    st.session_state.comparison_df = None
if 'raw_scoring_data' not in st.session_state:
    st.session_state.raw_scoring_data = None
if 'scoring_dataset' not in st.session_state:
    st.session_state.scoring_dataset = None
if 'relativities_df' not in st.session_state:
    st.session_state.relativities_df = None
if 'upload_config' not in st.session_state:
    st.session_state.upload_config = {}

# --- Comparison Mode Selection ---
st.header("Comparison Mode")
comparison_mode = st.radio(
    "Choose comparison method:",
    ["Compare Saved Models", "Upload Relativities & Dataset"],
    horizontal=True,
    help="Compare saved GLM models or upload external relativities with dataset",
    index=0 if st.session_state.comparison_mode == "Compare Saved Models" else 1
)

# Update session state when mode changes
if comparison_mode != st.session_state.comparison_mode:
    st.session_state.comparison_mode = comparison_mode
    # Clear results when mode changes
    st.session_state.comparison_results = None
    st.session_state.comparison_df = None
    st.session_state.pred_data = {}
    st.session_state.model_names = {}
    st.session_state.relativities_df = None
    st.session_state.upload_config = {}

if comparison_mode == "Compare Saved Models":
    # --- Check for saved models ---
    MODEL_DIR = "saved_models"
    Path(MODEL_DIR).mkdir(exist_ok=True)
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

    if len(model_files) < 2:
        st.warning("⚠️ You need at least two saved models to use this page. Please fit and save models using the 'GLM Fit' and 'Model Manager' pages.")
        st.stop()

    # --- Model Selection ---
    st.header("1. Select Models to Compare")
    col1, col2 = st.columns(2)

    with col1:
        model_name_1 = st.selectbox(
            "Select Model A (Base)",
            options=model_files,
            format_func=lambda x: x.replace(".pkl", ""),
            key="comp_model1"
        )

    options_2 = [f for f in model_files if f != model_name_1]
    if not options_2:
        st.error("❌ You only have one model saved. Please save a second, different model to compare.")
        st.stop()
        
    with col2:
        model_name_2 = st.selectbox(
            "Select Model B (Challenger)",
            options=options_2,
            format_func=lambda x: x.replace(".pkl", ""),
            key="comp_model2"
        )

    if model_name_1 == model_name_2:
        st.error("❌ Please select two different models for comparison.")
        st.stop()

    # --- Load Models ---
    @st.cache_data
    def load_model_bundle(filename):
        """Loads a model bundle from a pickle file."""
        with open(Path(MODEL_DIR) / filename, "rb") as f:
            bundle = pickle.load(f)
        return bundle

    bundle1 = load_model_bundle(model_name_1)
    bundle2 = load_model_bundle(model_name_2)

    model1 = bundle1['model']
    model2 = bundle2['model']
    name1 = model_name_1.replace(".pkl", "")
    name2 = model_name_2.replace(".pkl", "")

    # --- High-Level Premium Impact Analysis ---
    st.header("2. High-Level Premium Impact Analysis")

    # --- Let user choose the dataset ---
    st.subheader("Select Scoring Dataset")
    dataset_choice = st.radio(
        "Which dataset should be used to compare the models' pricing?",
        (f"{name1} (original data)", f"{name2} (original data)"),
        horizontal=True,
        help="This dataset will be used as the 'book of business' to see the overall financial impact of switching from one model to another."
    )

    if dataset_choice == f"{name1} (original data)":
        raw_scoring_data = bundle1['df_raw']
        col_map_to_use = bundle1['col_map']
        chosen_name = name1
    else:
        raw_scoring_data = bundle2['df_raw']
        col_map_to_use = bundle2['col_map']
        chosen_name = name2

    # Rename columns to the sanitized versions the model expects.
    raw_scoring_data = raw_scoring_data.rename(columns=col_map_to_use)

    st.info(f"ℹ️ Comparing models using the **{chosen_name}** dataset ({len(raw_scoring_data)} records). The data will be automatically prepared to be compatible with both models.")
    # --- Prepare the data and run predictions ---
    # Use the new utility function to make the data compatible with both models
    scoring_data = unify_and_prepare_dataset(raw_scoring_data, model1, model2)

    # Now predictions should work reliably
    pred_a = model1.predict(scoring_data)
    pred_b = model2.predict(scoring_data)

    # Create comparison dataframe for relativities
    comparison_df = extract_relativities(model1.params, model2.params, name1, name2)

    # Store predictions and data in session state
    st.session_state.pred_data = {
        'pred_a': pred_a,
        'pred_b': pred_b
    }
    st.session_state.model_names = {
        'name1': name1,
        'name2': name2,
        'model_a_col_name': name1,
        'model_b_col_name': name2
    }
    st.session_state.comparison_df = comparison_df
    st.session_state.raw_scoring_data = raw_scoring_data
    st.session_state.comparison_results = True

    # Store predictions for later use
    name1, name2 = name1, name2
    
    # Set original column names for saved models mode (for consistent display)
    model_a_col_name = name1
    model_b_col_name = name2
    
else:  # Upload Relativities & Dataset mode
    # Check if we have stored results to display
    if st.session_state.comparison_results and st.session_state.comparison_mode == "Upload Relativities & Dataset":
        # Add button to start over
        if st.button("🔄 Upload Different Files"):
            st.session_state.comparison_results = None
            st.session_state.comparison_df = None
            st.session_state.pred_data = {}
            st.session_state.model_names = {}
            st.session_state.relativities_df = None
            st.session_state.upload_config = {}
            st.session_state.scoring_dataset = None
            st.rerun()
    
    else:
        # Show upload interface
        # --- Upload and Configure Section ---
        st.header("1. Upload Relativities and Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Upload Relativities")
        relativities_file = st.file_uploader(
            "Upload relativities CSV",
            type=['csv'],
            help="CSV with 3 columns: 1) Factors (patsy format), 2) Model A relativities, 3) Model B relativities. Column names don't matter - only position. Formula: Premium = Exposure × Base Rate × Relativities"
        )
        
        if relativities_file:
            try:
                # Try UTF-8 first, then fallback to other encodings
                try:
                    relativities_df = pd.read_csv(relativities_file, encoding='utf-8')
                except UnicodeDecodeError:
                    # Reset file pointer and try with different encoding
                    relativities_file.seek(0)
                    try:
                        relativities_df = pd.read_csv(relativities_file, encoding='latin-1')
                    except UnicodeDecodeError:
                        # Reset file pointer and try with Windows encoding
                        relativities_file.seek(0)
                        relativities_df = pd.read_csv(relativities_file, encoding='cp1252')
                
                # Store original column names for display purposes
                if len(relativities_df.columns) >= 3:
                    original_col_names = list(relativities_df.columns)
                    factor_col_name = original_col_names[0]
                    model_a_col_name = original_col_names[1]
                    model_b_col_name = original_col_names[2]
                    
                    # Standardize column names for processing but keep originals for display
                    relativities_df.columns = ['Factor', 'Model_A_Values', 'Model_B_Values'] + list(relativities_df.columns[3:])
                    
                    # Store in session state
                    st.session_state.relativities_df = relativities_df
                    st.session_state.upload_config['model_a_col_name'] = model_a_col_name
                    st.session_state.upload_config['model_b_col_name'] = model_b_col_name
                    st.session_state.upload_config['factor_col_name'] = factor_col_name
                    
                    st.success(f"✅ Loaded {len(relativities_df)} relativities")
                    st.info(f"📋 Using columns: '{factor_col_name}' (factors), '{model_a_col_name}' (Model A), '{model_b_col_name}' (Model B)")
                else:
                    st.error("❌ CSV must have at least 3 columns: Factor, Model A values, Model B values")
                    st.stop()
                
                # Show preview
                with st.expander("Preview Relativities"):
                    st.dataframe(relativities_df.head(10))
                    
            except Exception as e:
                st.error(f"❌ Error reading relativities file: {str(e)}")
                st.info("💡 Try saving your CSV file with UTF-8 encoding, or ensure it doesn't contain special characters.")
    
    with col2:
        st.subheader("📋 Upload Dataset")
        dataset_file = st.file_uploader(
            "Upload scoring dataset CSV",
            type=['csv'],
            help="Dataset containing the portfolio to score"
        )
        
        if dataset_file:
            try:
                # Try UTF-8 first, then fallback to other encodings
                try:
                    scoring_dataset = pd.read_csv(dataset_file, encoding='utf-8')
                except UnicodeDecodeError:
                    # Reset file pointer and try with different encoding
                    dataset_file.seek(0)
                    try:
                        scoring_dataset = pd.read_csv(dataset_file, encoding='latin-1')
                    except UnicodeDecodeError:
                        # Reset file pointer and try with Windows encoding
                        dataset_file.seek(0)
                        scoring_dataset = pd.read_csv(dataset_file, encoding='cp1252')
                
                st.success(f"✅ Loaded {len(scoring_dataset)} records")
                
                # Store in session state
                st.session_state.scoring_dataset = scoring_dataset
                
                # Show preview of original data
                with st.expander("Preview Dataset (Original)"):
                    st.dataframe(scoring_dataset.head(5))
                    
            except Exception as e:
                st.error(f"❌ Error reading dataset file: {str(e)}")
                st.info("💡 Try saving your CSV file with UTF-8 encoding, or ensure it doesn't contain special characters.")
    
    # Configuration inputs - show only if dataset is loaded
    if st.session_state.scoring_dataset is not None:
        scoring_dataset = st.session_state.scoring_dataset
        st.subheader("⚙️ Pricing Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            base_rate = st.number_input(
                "Base Rate (per unit of exposure)",
                min_value=0.0,
                value=st.session_state.upload_config.get('base_rate', 0.001),
                step=0.0001,
                format="%.4f",
                help="The base rate per unit of exposure (e.g., per $1 of Sum Insured)"
            )
            st.session_state.upload_config['base_rate'] = base_rate
        
        with col2:
            # Get numeric columns for exposure selection
            numeric_columns = scoring_dataset.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                default_exposure_idx = 0
                if 'exposure_column' in st.session_state.upload_config and st.session_state.upload_config['exposure_column'] in numeric_columns:
                    default_exposure_idx = numeric_columns.index(st.session_state.upload_config['exposure_column'])
                
                exposure_column = st.selectbox(
                    "Exposure Column",
                    options=numeric_columns,
                    index=default_exposure_idx,
                    help="Select the exposure column (e.g., Sum_Insured)"
                )
                st.session_state.upload_config['exposure_column'] = exposure_column
            else:
                st.error("❌ No numeric columns found in dataset for exposure!")
                exposure_column = None
                st.stop()
    else:
        # Default values when no dataset is loaded yet
        base_rate = 0.001
        exposure_column = None
    
    # Proceed only if both files are uploaded and loaded successfully, and exposure column is selected
    if (st.session_state.relativities_df is not None and st.session_state.scoring_dataset is not None 
        and exposure_column is not None):
        
        # Get data from session state
        relativities_df = st.session_state.relativities_df
        scoring_dataset = st.session_state.scoring_dataset
        model_a_col_name = st.session_state.upload_config['model_a_col_name']
        model_b_col_name = st.session_state.upload_config['model_b_col_name']

        # --- Sanitize dataset column names to match factor names ---
        st.info("🔧 Sanitizing dataset column names to match factor format...")
        
        # Create column mapping for sanitization
        original_columns = scoring_dataset.columns.tolist()
        sanitized_columns = [sanitize_name(col) for col in original_columns]
        column_mapping = dict(zip(original_columns, sanitized_columns))
        
        # Apply sanitization
        scoring_dataset_sanitized = scoring_dataset.rename(columns=column_mapping)
        
        # Update exposure column name if it was sanitized
        original_exposure_column = exposure_column
        sanitized_exposure_column = sanitize_name(exposure_column)
        
        # Show column mapping
        with st.expander("📋 Column Name Mapping (Original → Sanitized)"):
            mapping_df = pd.DataFrame({
                'Original Column': original_columns,
                'Sanitized Column': sanitized_columns
            })
            st.dataframe(mapping_df)
            
            if original_exposure_column != sanitized_exposure_column:
                st.info(f"🔄 Exposure column '{original_exposure_column}' → '{sanitized_exposure_column}'")
        
        # Show preview of sanitized dataset
        with st.expander("Preview Dataset (Sanitized)"):
            st.dataframe(scoring_dataset_sanitized.head(5))
        
        # --- Process relativities and calculate premiums ---
        
        # Apply relativities
        try:
            scored_data = apply_relativities_to_dataset(
                scoring_dataset_sanitized, relativities_df, base_rate, sanitized_exposure_column
            )
            
            # Extract predictions
            pred_a = scored_data['Premium_A'].values
            pred_b = scored_data['Premium_B'].values
            
            # Set model names for consistency with saved models path using original column names
            name1 = model_a_col_name
            name2 = model_b_col_name
            
            # Store data in session state
            st.session_state.pred_data = {
                'pred_a': pred_a,
                'pred_b': pred_b
            }
            st.session_state.model_names = {
                'name1': name1,
                'name2': name2,
                'model_a_col_name': model_a_col_name,
                'model_b_col_name': model_b_col_name
            }
            st.session_state.scoring_dataset = scoring_dataset
            st.session_state.comparison_results = True
            
            # Prepare comparison dataframe using the utility function and original column names
            comparison_df = relativities_df.copy()
            comparison_df = comparison_df.rename(columns={
                'Factor': 'Factor',
                'Model_A_Values': name1,
                'Model_B_Values': name2
            })
            comparison_df['Ratio (B/A)'] = comparison_df[name2] / comparison_df[name1]
            
            # Add Level and Factor Type columns using the utility function
            parsed_info = comparison_df['Factor'].apply(parse_factor_name)
            comparison_df['Level'] = [info[1] for info in parsed_info]
            comparison_df['Factor Type'] = [info[2] for info in parsed_info]
            comparison_df['Factor'] = [info[0] for info in parsed_info]  # Use the cleaned factor name
            
            # Store comparison_df in session state
            st.session_state.comparison_df = comparison_df
            
        except Exception as e:
            st.error(f"❌ Error applying relativities: {str(e)}")
            st.stop()
            
    else:
        if st.session_state.relativities_df is None or st.session_state.scoring_dataset is None:
            st.info("👆 Please upload both relativities CSV and dataset CSV to proceed.")
        elif exposure_column is None:
            st.info("📋 Please select an exposure column from your dataset.")
        st.stop()

# Display Relativities Comparison for Upload mode after processing
if (st.session_state.comparison_mode == "Upload Relativities & Dataset" and 
    st.session_state.comparison_results and st.session_state.comparison_df is not None):
    
    st.header("2. Relativities Comparison")
    
    # Get stored data
    comparison_df = st.session_state.comparison_df
    name1 = st.session_state.model_names['name1']
    name2 = st.session_state.model_names['name2']
    
    st.markdown(f"Comparing **{name1}** vs. **{name2}**. The ratio shows how Model B's price changes relative to Model A for each factor.")
    
    # Create custom styling for the ratio column
    def color_ratio(val):
        """Color ratios: green for <100%, orange for >100%"""
        if val > 1.0:
            # Orange scale for values above 100%
            intensity = min((val - 1.0) * 2, 1.0)  # Scale intensity
            return f'background-color: rgba(255, 165, 0, {intensity * 0.7})'
        else:
            # Green scale for values below 100%
            intensity = min((1.0 - val) * 2, 1.0)  # Scale intensity
            return f'background-color: rgba(0, 128, 0, {intensity * 0.7})'
    
    st.dataframe(
        comparison_df.style.format({
            name1: '{:.3f}',
            name2: '{:.3f}',
            'Ratio (B/A)': '{:.2%}'
        }).map(color_ratio, subset=['Ratio (B/A)']),
        use_container_width=True,
        hide_index=True,
        column_config={"Factor": st.column_config.Column(pinned=True)}
    )

# --- Common sections for both modes (Premium Impact Analysis) ---
# Check if we have results to display (either from current session or session state)
if st.session_state.comparison_results and st.session_state.pred_data:
    st.header("3. High-Level Premium Impact Analysis")

    # Get data from session state
    pred_a = st.session_state.pred_data['pred_a']
    pred_b = st.session_state.pred_data['pred_b']
    name1 = st.session_state.model_names['name1']
    name2 = st.session_state.model_names['name2']
    model_a_col_name = st.session_state.model_names['model_a_col_name']
    model_b_col_name = st.session_state.model_names['model_b_col_name']
    comparison_df = st.session_state.comparison_df
    
    # Get dataset for Excel export
    if st.session_state.comparison_mode == "Compare Saved Models":
        dataset_for_export = st.session_state.raw_scoring_data
    else:
        dataset_for_export = st.session_state.scoring_dataset

    # --- Display Results ---
    st.subheader("Overall Summary")
    impact_df = pd.DataFrame({
        f'Predicted_Premium_{name1}': pred_a,
        f'Predicted_Premium_{name2}': pred_b
    })
    impact_df['Ratio (B/A)'] = impact_df[f'Predicted_Premium_{name2}'] / impact_df[f'Predicted_Premium_{name1}']
    impact_df['Difference (B-A)'] = impact_df[f'Predicted_Premium_{name2}'] - impact_df[f'Predicted_Premium_{name1}']

    avg_a = impact_df[f'Predicted_Premium_{name1}'].mean()
    avg_b = impact_df[f'Predicted_Premium_{name2}'].mean()
    avg_ratio = (avg_b / avg_a - 1) if avg_a != 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Avg. Premium ({name1})", f"${avg_a:,.2f}")
    col2.metric(f"Avg. Premium ({name2})", f"${avg_b:,.2f}", f"{avg_ratio:.2%}")
    col3.metric("Median Premium Ratio", f"{impact_df['Ratio (B/A)'].median():.3f}x", help="The median policy is this many times more/less expensive under Model B vs. Model A.")

    st.subheader("Distribution of Premium Ratios (Model B / Model A)")

    fig_premium, ax = plt.subplots()
    ax.hist(impact_df['Ratio (B/A)'], bins=80, range=(0.5, 2.0), color='mediumseagreen', alpha=0.7)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='No Change (Ratio = 1.0)')
    ax.set_title("Distribution of Premium Ratios", fontsize=16)
    ax.set_xlabel("Ratio (Premium B / Premium A)")
    ax.set_ylabel("Number of Policies")
    ax.legend()
    st.pyplot(fig_premium)

    with st.expander("View Full Impact Data"):
        st.dataframe(impact_df.style.format({
            f'Predicted_Premium_{name1}': "${:,.2f}",
            f'Predicted_Premium_{name2}': "${:,.2f}",
            'Ratio (B/A)': "{:.2f}x",
            'Difference (B-A)': "${:,.2f}"
        }))

    # --- Visualize Comparison ---
    st.header("4. Visual Comparison")

    vis_type = st.radio(
        "Select comparison type",
        ["Categorical Factors", "Numeric Factors"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # --- Categorical Visualization ---
    if vis_type == "Categorical Factors":
        st.subheader("Categorical Factor Relativities")
        categorical_factors = sorted(list(
            st.session_state.comparison_df[st.session_state.comparison_df['Factor Type'] == 'Categorical']['Factor'].unique()
        ))

        if not categorical_factors:
            st.info("ℹ️ No common categorical factors were found between the two models to visualize.")
        else:
            selected_factor = st.selectbox(
                "Select a factor to visualize",
                options=categorical_factors
            )

            if selected_factor:
                plot_data = st.session_state.comparison_df[st.session_state.comparison_df['Factor'] == selected_factor].copy()
                
                # --- For uploaded relativities mode, handle base level differently ---
                if st.session_state.comparison_mode == "Upload Relativities & Dataset":
                    base_level_name = "Base"
                    # Look for 'Other' entries which represent base levels for the same factor
                    original_factor_name = None
                    for _, row in st.session_state.relativities_df.iterrows():
                        factor_col, level, ftype = parse_factor_name(row['Factor'])
                        if factor_col == selected_factor and level == 'Other':
                            base_row = pd.DataFrame([{
                                'Factor': selected_factor,
                                'Level': base_level_name,
                                name1: row['Model_A_Values'],
                                name2: row['Model_B_Values'],
                            }])
                            break
                    else:
                        # No 'Other' entry found, use 1.0 as base
                        base_row = pd.DataFrame([{
                            'Factor': selected_factor, 
                            'Level': base_level_name,
                            name1: 1.0, 
                            name2: 1.0, 
                        }])
                else:
                    # --- Robustly find the base level name for saved models ---
                    base_level_name = "Base" # Default
                    try:
                        # Try finding the factor info first with C() wrapper, then without
                        factor_info = model1.model.data.design_info.factor_infos.get(f'C({selected_factor})')
                        if not factor_info:
                            factor_info = model1.model.data.design_info.factor_infos.get(selected_factor)
                        
                        if factor_info:
                            base_level_name = f"Base ({factor_info.categories[0]})"
                    except (KeyError, AttributeError):
                        st.warning(f"Could not determine the base level for '{selected_factor}'. Defaulting to 'Base'.")
                    
                    # Add the base level row for a complete visualization
                    base_row = pd.DataFrame([{
                        'Factor': selected_factor, 
                        'Level': base_level_name,
                        name1: 1.0, 
                        name2: 1.0, 
                    }])
                
                plot_data = pd.concat([base_row, plot_data], ignore_index=True)
                plot_data.set_index('Level', inplace=True)

                fig = plot_relativity_comparison(plot_data, selected_factor, name1, name2)
                st.pyplot(fig)

    # --- Numeric Visualization ---
    elif vis_type == "Numeric Factors":
        st.subheader("Numeric Factor Relativities")
        numeric_data = st.session_state.comparison_df[st.session_state.comparison_df['Factor Type'] == 'Continuous'].copy()

        if numeric_data.empty:
            st.info("ℹ️ No common numeric factors were found between the two models to visualize.")
        else:
            if st.session_state.comparison_mode == "Compare Saved Models":
                fig = plot_numeric_factor_comparison(numeric_data, name1, name2)
                st.pyplot(fig)
            else:
                st.info("ℹ️ Numeric factor visualization is not supported for uploaded relativities mode (continuous factors require coefficient application).")

    # --- Section 5: Download Report ---
    st.header("5. Download Comparison Report")

    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # --- Create and format the Impact Summary sheet ---
            workbook = writer.book
            summary_sheet = workbook.add_worksheet("Impact Summary")

            # Define formats
            header_format = workbook.add_format({'bold': True, 'valign': 'top', 'fg_color': '#DDEBF7', 'border': 1})
            metric_header_format = workbook.add_format({'bold': True, 'align': 'left'})
            currency_format = workbook.add_format({'num_format': '$#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            ratio_format = workbook.add_format({'num_format': '0.000"x"'})

            # Set column widths
            summary_sheet.set_column('A:A', 25)
            summary_sheet.set_column('B:D', 20)

            # Write table headers and data
            headers = ['Metric', f'{model_a_col_name}', f'{model_b_col_name}', f'Comparison ({model_b_col_name} vs {model_a_col_name})']
            summary_sheet.write_row('A1', headers, header_format)
            summary_sheet.write('A2', 'Average Premium', metric_header_format)
            summary_sheet.write('B2', avg_a, currency_format)
            summary_sheet.write('C2', avg_b, currency_format)
            summary_sheet.write('D2', avg_ratio, percent_format)
            summary_sheet.write('A3', 'Median Premium Ratio', metric_header_format)
            summary_sheet.write('D3', impact_df['Ratio (B/A)'].median(), ratio_format)

            # --- NEW: Save the chart and insert it into the sheet ---
            if 'fig_premium' in locals() or 'fig_premium' in globals():
                img_data = BytesIO()
                fig_premium.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
                img_data.seek(0)
                
                # Place the image below the summary table
                summary_sheet.insert_image('A5', 'impact_ratio_chart.png', {'image_data': img_data})

            # --- Write the other dataframes to separate sheets ---
            st.session_state.comparison_df.to_excel(writer, sheet_name="Relativity Comparison", index=False)
            
            # Create detailed predictions sheet with full dataset + relativities + predictions
            detailed_df = dataset_for_export.copy()
            
            # Add relativity columns for each rating factor
            if st.session_state.comparison_mode == "Compare Saved Models":
                # For saved models, apply relativities to the dataset
                try:
                    # Apply relativities to get factor-level relativities for each record
                    relativities_audit_df = apply_relativities_to_dataset_for_audit(
                        detailed_df.rename(columns={v: k for k, v in col_map_to_use.items()}),  # Convert back to original column names
                        comparison_df, 
                        model1, 
                        model2,
                        name1,
                        name2
                    )
                    
                    # Add the relativity columns to the detailed dataframe
                    for col in relativities_audit_df.columns:
                        if col not in detailed_df.columns:
                            detailed_df[col] = relativities_audit_df[col]
                            
                except Exception as e:
                    st.warning(f"Could not add relativity audit columns: {str(e)}")
            
            else:  # Upload Relativities & Dataset mode
                # For uploaded relativities, apply the relativities directly
                try:
                    relativities_audit_df = apply_relativities_to_dataset_for_audit_upload(
                        detailed_df,
                        st.session_state.relativities_df,
                        name1,
                        name2
                    )
                    
                    # Add the relativity columns to the detailed dataframe
                    for col in relativities_audit_df.columns:
                        if col not in detailed_df.columns:
                            detailed_df[col] = relativities_audit_df[col]
                            
                except Exception as e:
                    st.warning(f"Could not add relativity audit columns: {str(e)}")
            
            # Add predictions to the right side
            detailed_df[f'Predicted_Premium_{model_a_col_name}'] = pred_a
            detailed_df[f'Predicted_Premium_{model_b_col_name}'] = pred_b
            detailed_df['Premium_Ratio_B_vs_A'] = pred_b / pred_a
            detailed_df['Premium_Difference_B_minus_A'] = pred_b - pred_a
            
            # Clean up relativity column names before writing to Excel
            column_rename_map = {}
            for col in detailed_df.columns:
                if col.startswith('Rel_') and col.endswith('_A'):
                    # Extract factor name and add A suffix
                    factor_name = col.replace('Rel_', '').replace('_A', '')
                    column_rename_map[col] = f'{factor_name}_A'
                elif col.startswith('Rel_') and col.endswith('_B'):
                    # Extract factor name and add B suffix
                    factor_name = col.replace('Rel_', '').replace('_B', '')
                    column_rename_map[col] = f'{factor_name}_B'
            
            # Apply the column name changes
            detailed_df_for_excel = detailed_df.rename(columns=column_rename_map)
            
            # Write to Excel with custom formatting and grouping
            detailed_df_for_excel.to_excel(writer, sheet_name="Detailed Predictions", index=False)
            
            # Add column grouping to the Detailed Predictions sheet
            detailed_sheet = writer.sheets["Detailed Predictions"]
            
            # Find relativity columns for grouping (using the renamed columns)
            model_a_rel_cols = [col for col in detailed_df_for_excel.columns if col.endswith('_A') and not col.startswith('Predicted_Premium_')]
            model_b_rel_cols = [col for col in detailed_df_for_excel.columns if col.endswith('_B') and not col.startswith('Predicted_Premium_')]
            
            # Find prediction columns for color coding (only the first two prediction columns)
            prediction_cols = [
                f'Predicted_Premium_{model_a_col_name}',
                f'Predicted_Premium_{model_b_col_name}'
            ]
            
            # Separate the summary/ratio columns (no color coding for these)
            summary_cols = [
                'Premium_Ratio_B_vs_A',
                'Premium_Difference_B_minus_A'
            ]
            
            if model_a_rel_cols or model_b_rel_cols or prediction_cols:
                # Get column indices (0-based) using the renamed dataframe
                all_columns = list(detailed_df_for_excel.columns)
                
                # Define header formats for different model colors
                header_format_model_a = workbook.add_format({
                    'bold': True, 
                    'valign': 'top', 
                    'fg_color': '#D5E4F7',  # Light blue background for Model A
                    'border': 1,
                    'text_wrap': True
                })
                
                header_format_model_b = workbook.add_format({
                    'bold': True, 
                    'valign': 'top', 
                    'fg_color': '#E2EFDA',  # Light green background for Model B
                    'border': 1,
                    'text_wrap': True
                })
                
                # Format for prediction/summary columns
                header_format_predictions = workbook.add_format({
                    'bold': True, 
                    'valign': 'top', 
                    'fg_color': '#FFF2CC',  # Light yellow background for predictions
                    'border': 1,
                    'text_wrap': True
                })
                
                # Set up grouping and formatting for Model A columns
                if model_a_rel_cols:
                    start_idx = all_columns.index(model_a_rel_cols[0])
                    end_idx = all_columns.index(model_a_rel_cols[-1])
                    detailed_sheet.set_column(start_idx, end_idx, 12)  # Set width
                    
                    # Create outline/grouping for Model A relativities
                    detailed_sheet.outline_settings(True, False, False, True)
                    for i in range(start_idx, end_idx + 1):
                        detailed_sheet.set_column(i, i, None, None, {'level': 1})
                    
                    # Apply Model A header formatting
                    for col_idx, col_name in enumerate(all_columns):
                        if col_name in model_a_rel_cols:
                            detailed_sheet.write(0, col_idx, col_name, header_format_model_a)
                
                # Set up grouping and formatting for Model B columns  
                if model_b_rel_cols:
                    start_idx = all_columns.index(model_b_rel_cols[0])
                    end_idx = all_columns.index(model_b_rel_cols[-1])
                    detailed_sheet.set_column(start_idx, end_idx, 12)  # Set width
                    
                    # Create outline/grouping for Model B relativities
                    for i in range(start_idx, end_idx + 1):
                        detailed_sheet.set_column(i, i, None, None, {'level': 1})
                    
                    # Apply Model B header formatting
                    for col_idx, col_name in enumerate(all_columns):
                        if col_name in model_b_rel_cols:
                            detailed_sheet.write(0, col_idx, col_name, header_format_model_b)
                
                # Color code prediction columns (only the first two)
                for col_idx, col_name in enumerate(all_columns):
                    if col_name in prediction_cols:
                        # Determine which color to use based on column content
                        if model_a_col_name in col_name:
                            detailed_sheet.write(0, col_idx, col_name, header_format_model_a)
                        elif model_b_col_name in col_name:
                            detailed_sheet.write(0, col_idx, col_name, header_format_model_b)
                
                # Add a summary row explaining the color coding
                summary_row = len(detailed_df_for_excel) + 3  # Add some space after data
                detailed_sheet.write(summary_row, 0, "Color Legend:", workbook.add_format({'bold': True}))
                detailed_sheet.write(summary_row + 1, 0, f"Model A ({name1}):", header_format_model_a)
                detailed_sheet.write(summary_row + 2, 0, f"Model B ({name2}):", header_format_model_b)
                detailed_sheet.write(summary_row + 3, 0, "Predictions/Summary:", header_format_predictions)

        # Create the download button
        st.download_button(
            label="📥 Download Excel Report",
            data=output.getvalue(),
            file_name=f"Pricing_Comparison_{model_a_col_name}_vs_{model_b_col_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

    ### AI Insights
    get_ai_insights()
    
else:
    # No results to display yet
    if st.session_state.comparison_mode == "Compare Saved Models":
        st.info("👆 Please select two models above to see the comparison results.")
    else:
        st.info("👆 Please upload relativities and dataset files above to see the comparison results.")