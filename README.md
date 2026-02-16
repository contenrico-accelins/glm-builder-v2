# GLM Builder

A Streamlit web application for building and analyzing Generalized Linear Models (GLMs) for insurance data analysis.

## Overview

GLM Builder is an interactive tool designed to help actuaries build, analyze, and deploy GLM models. The application provides a user-friendly interface for data exploration, model fitting, and pricing analysis.

## Features

### 🏠 Home Page
- **Data Upload**: Upload CSV datasets for analysis
- **Data Preview**: View and inspect uploaded data
- **Variable Selection**: Choose response variables and predictors
- **Distribution Selection**: Select from Gamma, Gaussian, Poisson, or Tweedie distributions
- **Tweedie Optimization**: Automatic optimization of Tweedie variance power parameter

### 📈 One-Way Analysis
- Univariate analysis of predictor variables
- Visual exploration of predictor effects
- Statistical summaries and distributions

### 📊 GLM Fit
- Model fitting with selected predictors and distribution
- Model diagnostics and residual analysis
- Statistical summaries and coefficient analysis
- Excel report generation

### 📁 Model Manager
- Save and load trained models
- Model comparison and version control
- Model metadata management

### ⚖️ Pricing Comparison
- Compare pricing across different models
- Scenario analysis and what-if pricing
- Rate comparison tools

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows, macOS, or Linux

### Setup

1. Clone the repository:
```bash
git clone https://github.com/accelins/glm-builder.git
cd glm-builder
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Application

To start the GLM Builder application, run the following command in your terminal:

```bash
python -m streamlit run 🏠_Home.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Workflow

1. **Upload Data**: Start on the Home page and upload your insurance CSV dataset
2. **Select Variables**: Choose your response variable (typically premium) and predictor variables
3. **Choose Distribution**: Select the appropriate GLM distribution for your data
4. **Analyze**: Use the One-Way Analysis page to explore individual predictors
5. **Fit Model**: Build your GLM on the GLM Fit page
6. **Save Model**: Use the Model Manager to save your fitted model
7. **Price**: Compare pricing scenarios using the Pricing Comparison page

### Data Requirements

Your CSV file should contain:
- A response variable (e.g., premium, claims)
- Predictor variables (both categorical and numeric)
- Clean, properly formatted data

## Technical Details

### Supported Distributions
- **Gamma**: For continuous, positive response variables (common for premium modeling)
- **Gaussian**: For normally distributed response variables
- **Poisson**: For count data
- **Tweedie**: For zero-inflated continuous data (common for claims modeling)

### Dependencies

The application is built using:
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Statsmodels**: Statistical modeling and GLM fitting
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Additional statistical tools

## File Structure

```
glm-builder/
├── 🏠_Home.py              # Main application entry point
├── utils.py                # Utility functions and helpers
├── requirements.txt        # Python dependencies
├── pages/                  # Streamlit pages
│   ├── 1_📈_One_Way_Analysis.py
│   ├── 2_📊_GLM_Fit.py
│   ├── 3_📁_Model_Manager.py
│   └── 4_⚖️_Pricing_Comparison.py
└── saved_models/           # Directory for saved model files
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is proprietary software developed by Accelerant Holdings.

## Support

For questions, issues, or feature requests, please contact the development team or create an issue in the repository.

## Version History

- **v1.0**: Initial release with core GLM functionality
- Data upload and preprocessing
- GLM fitting with multiple distributions
- Model management and comparison
- Pricing analysis tools
