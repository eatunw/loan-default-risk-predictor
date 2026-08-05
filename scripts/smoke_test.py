import json
import joblib
import numpy as np
import os

print('Smoke test started')
base = os.getcwd()
print('CWD:', base)

# Load artifacts
with open('feature_names.json','r') as f:
    feature_names = json.load(f)

print('feature_names loaded:', len(feature_names))
scaler = joblib.load('scaler.pkl')
print('scaler loaded')
xgb = joblib.load('loan_default_xgb.pkl')
print('xgb loaded')

# Keras removed: running XGBoost-only smoke test
keras_model = None
print('Keras/TensorFlow removed; running XGBoost-only smoke test')

# Default numeric values (match app defaults)
defaults = {
    'loan_amnt': 15000.0,
    'term': 36,
    'int_rate': 12.0,
    'installment': 450.0,
    'annual_inc': 75000.0,
    'dti': 18.0,
    'earliest_cr_line': 2000,
    'open_acc': 10,
    'pub_rec': 0,
    'revol_bal': 15000.0,
    'revol_util': 45.0,
    'total_acc': 25,
    'mort_acc': 1,
    'pub_rec_bankruptcies': 0,
}

# Build input vector in scaler order if available
if hasattr(scaler, 'feature_names_in_'):
    expected = list(scaler.feature_names_in_)
else:
    expected = feature_names

# Start with zeros
row = {c: 0 for c in expected}
# Fill numeric defaults where names match
for k, v in defaults.items():
    if k in row:
        row[k] = v

# Pick a sub-grade present in feature_names
sub_grade = None
for sg in ['B4','B3','B5','A4']:
    if sg in expected:
        sub_grade = sg
        break
if sub_grade:
    row[sub_grade] = 1

# Attempt simple category fills for a few likely columns
for col in expected:
    if col.startswith('application_type_') and 'application_type_INDIVIDUAL' in expected:
        row['application_type_INDIVIDUAL'] = 1
    if col.startswith('verification_status_') and 'verification_status_Not Verified' in expected:
        row['verification_status_Not Verified'] = 1

# Create numpy array for scaler
import pandas as pd
input_df = pd.DataFrame([row], columns=expected)

# Scale
try:
    X_scaled = scaler.transform(input_df)
    print('scaler.transform OK')
except Exception as e:
    print('scaler.transform failed:', type(e).__name__, e)
    raise

# Predict XGB
try:
    prob = xgb.predict_proba(X_scaled)[0,1]
    print('XGBoost prediction probability:', prob)
except Exception as e:
    print('XGBoost predict failed:', type(e).__name__, e)

# Predict Keras if available
if keras_model is not None:
    try:
        p = keras_model.predict(X_scaled, verbose=0)[0,0]
        print('Keras prediction probability:', float(p))
    except Exception as e:
        print('Keras predict failed:', type(e).__name__, e)

print('Smoke test finished')
