# Employee Attrition Predictor

Employee Attrition Predictor is a Python Streamlit web application that uses machine learning to predict whether an employee is at risk of leaving the company. The application provides both manual input for single employee predictions and CSV upload for batch predictions.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup
- **Python Version**: Requires Python 3.12+ (tested with Python 3.12.3)
- **Install Dependencies**: `pip3 install -r requirements.txt` -- takes 45 seconds to complete. NEVER CANCEL. Set timeout to 120+ seconds.
- **Validate Installation**: `python3 -c "import streamlit, pandas, joblib, plotly; print('All dependencies installed successfully')"` -- takes 5 seconds

### Build and Run
- **Start Application**: `streamlit run app.py --server.port 8501 --server.address 0.0.0.0` -- starts in 10 seconds. NEVER CANCEL during startup.
- **Access Application**: Navigate to `http://localhost:8501` in browser
- **Stop Application**: Use Ctrl+C or stop the process

### Testing and Validation
- **Syntax Check**: `python3 -m py_compile app.py` -- takes 2 seconds
- **Model Test**: Run the following validation script (takes 5 seconds):
```python
python3 -c "
import pandas as pd
import joblib
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')
print('Model artifacts loaded successfully')
print(f'Model expects {len(selected_features)} features')
print('Model validation: PASSED')
"
```

## Validation Scenarios

### ALWAYS Test These Scenarios After Making Changes:
1. **Manual Input Prediction**:
   - Start the application: `streamlit run app.py`
   - Open browser to `http://localhost:8501`
   - Use default values in Manual Input mode
   - Click "🔮 Predict Attrition Risk" button
   - Verify prediction result appears with probability percentage
   - Verify chart visualization displays correctly

2. **CSV Upload Prediction**:
   - Switch to "Upload CSV" mode in sidebar
   - Create test CSV with required columns:
     ```csv
     StockOptionLevel,JobInvolvement,JobSatisfaction,YearsWithCurrManager,EnvironmentSatisfaction,YearsInCurrentRole,TotalWorkingYears,Age,YearsAtCompany,JobLevel,MonthlyIncome
     1,3,3,5,3,2,10,35,5,2,5000
     ```
   - Upload the test file
   - Verify batch predictions appear in table format
   - Verify probability distribution chart displays

3. **Model Components Validation**:
   - Verify all required files exist: `app.py`, `logistic_regression_model.pkl`, `scaler.pkl`, `selected_features.pkl`, `requirements.txt`
   - Test model loading without errors

## Application Architecture

### Core Files
- **`app.py`** - Main Streamlit application (188 lines)
- **`requirements.txt`** - Python dependencies (6 packages)
- **`logistic_regression_model.pkl`** - Pre-trained machine learning model
- **`scaler.pkl`** - MinMaxScaler for feature normalization
- **`selected_features.pkl`** - List of expected model features (20 features)

### Dependencies
```
streamlit - Web application framework
scikit-learn - Machine learning library
pandas - Data manipulation
numpy - Numerical computing
imbalanced-learn - Handling imbalanced datasets
plotly - Interactive visualizations
```

### Model Features
The model expects these 20 features (some are one-hot encoded):
- Numeric: StockOptionLevel, JobInvolvement, JobSatisfaction, YearsWithCurrManager, EnvironmentSatisfaction, YearsInCurrentRole, TotalWorkingYears, Age, YearsAtCompany, JobLevel, MonthlyIncome
- Categorical (encoded): MaritalStatus_Married, Department_Research & Development, EducationField_*, JobRole_*, Gender_Male, BusinessTravel_Travel_Rarely

## Common Tasks

### Starting Fresh Development Session
1. `cd /path/to/AttritionPredictor`
2. `pip3 install -r requirements.txt` (45 seconds, don't cancel)
3. `python3 -m py_compile app.py` (syntax check)
4. `streamlit run app.py` (start application)
5. Test both manual input and CSV upload modes

### Making Code Changes
1. Always test syntax: `python3 -m py_compile app.py`
2. Start application and manually test both modes
3. Verify model predictions work correctly
4. Check browser console for JavaScript errors

### Troubleshooting
- **Import Errors**: Re-run `pip3 install -r requirements.txt`
- **Model Loading Errors**: Verify all `.pkl` files exist and are not corrupted
- **Streamlit Warnings**: Normal when importing app.py outside streamlit context
- **Browser Access Issues**: Ensure application started with `--server.address 0.0.0.0`

## Repository Structure
```
.
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── logistic_regression_model.pkl   # Trained ML model
├── scaler.pkl                     # Feature scaler
├── selected_features.pkl          # Expected features list
└── .github/
    └── copilot-instructions.md    # This file
```

## Performance Notes
- **Startup Time**: Application starts in ~10 seconds
- **Dependency Installation**: ~45 seconds for all packages
- **Prediction Time**: <1 second for single prediction
- **CSV Processing**: Depends on file size, typically <5 seconds for small files
- **Memory Usage**: ~200MB for loaded models and dependencies

## Validation Checklist
When making changes, always verify:
- [ ] `python3 -m py_compile app.py` passes
- [ ] Application starts without errors
- [ ] Manual prediction works with default values
- [ ] CSV upload accepts valid test file
- [ ] Predictions return reasonable probability values (0-100%)
- [ ] Charts and visualizations display correctly
- [ ] No browser console errors

Never skip these validation steps - they ensure the application remains functional for end users.