"""
Loan Default Risk Predictor - Streamlit Application
Professional UI for predicting loan default risk using ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #1e3a8a;
        --primary-light: #3b82f6;
        --secondary: #064e3b;
        --accent: #f59e0b;
        --danger: #dc2626;
        --success: #16a34a;
        --bg-light: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border: #e2e8f0;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 40px rgba(30, 58, 138, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .main-header p {
        margin: 0.5rem 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Cards */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }

    .risk-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }

    .risk-high { border-left: 5px solid var(--danger); }
    .risk-medium { border-left: 5px solid var(--accent); }
    .risk-low { border-left: 5px solid var(--success); }

    /* Input sections */
    .input-section {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Gauge chart container */
    .gauge-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem;
    }

    /* Feature importance bar */
    .feature-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(30, 58, 138, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Sidebar */
    .css-1d391kg {
        background: var(--bg-light);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: 1px solid var(--border);
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary);
        color: white;
        border-color: var(--primary);
    }

    /* Input labels */
    .stNumberInput label, .stSelectbox label, .stSlider label {
        font-weight: 500 !important;
        color: var(--text-primary) !important;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Risk probability text */
    .prob-text {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
    }

    .prob-label {
        text-align: center;
        color: var(--text-secondary);
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    /* Feature importance */
    .feature-importance-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 0;
    }

    .feature-name {
        min-width: 200px;
        font-weight: 500;
    }

    .feature-bar-container {
        flex: 1;
        height: 24px;
        background: var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    .feature-value {
        min-width: 60px;
        text-align: right;
        font-weight: 600;
        color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODELS AND ARTIFACTS ====================

@st.cache_resource
def load_artifacts():
    """Load all models, scaler, and feature names"""
    artifacts = {}

    # Load feature names
    with open('feature_names.json', 'r') as f:
        artifacts['feature_names'] = json.load(f)

    # Load scaler
    artifacts['scaler'] = joblib.load('scaler.pkl')

    # Load XGBoost model
    artifacts['xgb_model'] = joblib.load('loan_default_xgb.pkl')

    # Load Keras model
    try:
        artifacts['keras_model'] = load_model('loan_default_model_v1.keras')
        artifacts['has_keras'] = True
    except:
        artifacts['has_keras'] = False

    return artifacts

artifacts = load_artifacts()
feature_names = artifacts['feature_names']
scaler = artifacts['scaler']
xgb_model = artifacts['xgb_model']
has_keras = artifacts['has_keras']
keras_model = artifacts['keras_model'] if has_keras else None

# Feature categories for UI organization
NUMERICAL_FEATURES = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti',
    'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'total_acc', 'mort_acc', 'pub_rec_bankruptcies'
]

SUB_GRADE_FEATURES = [f for f in feature_names if f in ['A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5']]

VERIFICATION_FEATURES = [f for f in feature_names if f.startswith('verification_status_')]
APPLICATION_FEATURES = [f for f in feature_names if f.startswith('application_type_')]
INITIAL_LIST_FEATURES = [f for f in feature_names if f.startswith('initial_list_status_')]
PURPOSE_FEATURES = [f for f in feature_names if f.startswith('purpose_')]
HOME_OWNERSHIP_FEATURES = [f for f in feature_names if f in ['OTHER', 'OWN', 'RENT']]

# Feature descriptions for tooltips
FEATURE_DESCRIPTIONS = {
    'loan_amnt': 'The listed amount of the loan applied for by the borrower',
    'term': 'Number of payments on the loan (36 or 60 months)',
    'int_rate': 'Interest rate on the loan (%)',
    'installment': 'Monthly payment owed by the borrower',
    'annual_inc': 'Annual income of the borrower',
    'dti': 'Debt-to-income ratio (%)',
    'earliest_cr_line': 'Year the borrower\'s earliest reported credit line was opened',
    'open_acc': 'Number of open credit lines',
    'pub_rec': 'Number of derogatory public records',
    'revol_bal': 'Total revolving credit balance',
    'revol_util': 'Revolving line utilization rate (%)',
    'total_acc': 'Total number of credit lines currently in credit file',
    'mort_acc': 'Number of mortgage accounts',
    'pub_rec_bankruptcies': 'Number of public record bankruptcies',
}

# Default values based on training data statistics
DEFAULT_VALUES = {
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

# ==================== HELPER FUNCTIONS ====================

def prepare_input_data(inputs_dict):
    """
    Prepares input dictionary into a DataFrame aligned with model expectations.
    The scaler was trained on ALL features (including one-hot encoded categoricals),
    so we must pass the complete feature set in the exact same order.
    """
    # 1. Convert the raw dictionary to a DataFrame
    input_df = pd.DataFrame([inputs_dict])

    # 2. Ensure all dummy/one-hot columns match the training structure exactly
    # (Assuming scaler.feature_names_in_ contains the exact columns expected)
    if hasattr(scaler, "feature_names_in_"):
        expected_features = scaler.feature_names_in_
        # Add any missing columns with 0
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        # Reorder columns to match the scaler's exact training order
        input_df = input_df[expected_features]

        # 3. Scale the entire DataFrame (since the scaler expects all features)
        scaled_values = scaler.transform(input_df)
        input_df = pd.DataFrame(scaled_values, columns=expected_features)
    else:
        # Fallback: scale only numerical columns if scaler has no feature_names_in_
        numerical_cols = [c for c in NUMERICAL_FEATURES if c in input_df.columns]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    return input_df

def predict_xgb(input_df):
    """Get XGBoost prediction"""
    prob = xgb_model.predict_proba(input_df)[0, 1]
    return prob

def predict_keras(input_df):
    """Get Keras model prediction"""
    if keras_model is None:
        return None
    prob = keras_model.predict(input_df, verbose=0)[0, 0]
    return float(prob)

def get_risk_level(prob):
    """Determine risk level based on probability"""
    if prob >= 0.7:
        return 'low', 'Low Risk', '🟢'
    elif prob >= 0.4:
        return 'medium', 'Medium Risk', '🟡'
    else:
        return 'high', 'High Risk', '🔴'

def create_gauge_chart(probability, title="Default Probability"):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#1e293b'}},
        delta={'reference': 50, 'increasing': {'color': "#dc2626"}, 'decreasing': {'color': "#16a34a"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': "#1e3a8a"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': '#dcfce7'},
                {'range': [30, 60], 'color': '#fef3c7'},
                {'range': [60, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1e293b', 'family': 'Inter, sans-serif'}
    )
    return fig

def create_risk_breakdown_chart(prob_xgb, prob_keras=None):
    """Create a comparison chart"""
    models = ['XGBoost']
    probs = [prob_xgb * 100]
    colors = ['#3b82f6']

    if prob_keras is not None:
        models.append('Neural Network')
        probs.append(prob_keras * 100)
        colors.append('#8b5cf6')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=probs,
        marker_color=colors,
        text=[f'{p:.1f}%' for p in probs],
        textposition='auto',
        textfont={'size': 16, 'color': 'white'},
        hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
    ))

    # Add threshold lines
    fig.add_hline(y=30, line_dash="dash", line_color="#16a34a", annotation_text="Low Risk Threshold (30%)")
    fig.add_hline(y=60, line_dash="dash", line_color="#dc2626", annotation_text="High Risk Threshold (60%)")

    fig.update_layout(
        title="Model Comparison: Default Probability",
        yaxis_title="Default Probability (%)",
        yaxis_range=[0, 100],
        height=350,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif'},
        showlegend=False
    )
    return fig

def create_feature_importance_chart(input_df):
    """Create feature importance visualization for the specific prediction"""
    # Get feature importances from XGBoost
    importances = xgb_model.feature_importances_

    # Get feature values for this prediction
    feature_values = input_df.iloc[0].values

    # Create dataframe
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Value': feature_values
    })

    # Filter to top 15 by importance
    fi_df = fi_df.nlargest(15, 'Importance')
    fi_df = fi_df.sort_values('Importance', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=fi_df['Feature'],
        x=fi_df['Importance'],
        orientation='h',
        marker_color='#3b82f6',
        hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title="Top 15 Feature Importances (XGBoost)",
        xaxis_title="Importance",
        height=450,
        margin=dict(l=150, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif'}
    )
    return fig

# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Loan Default Risk Predictor</h1>
        <p>AI-powered credit risk assessment for lending decisions</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar - Model selection and info
    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")

        model_options = ['XGBoost (Recommended)']
        if has_keras:
            model_options.append('Neural Network (Keras)')
            model_options.append('Ensemble (Both)')

        selected_model = st.selectbox(
            "Select Model",
            model_options,
            help="XGBoost generally performs better on tabular data. Ensemble averages both models."
        )

        st.markdown("---")
        st.markdown("### 📊 Model Performance")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("XGBoost Accuracy", "75%", "↑ 3%")
            st.metric("XGBoost F1 (Macro)", "0.64", "↑ 0.04")
        with col2:
            if has_keras:
                st.metric("NN Accuracy", "69%", "↑ 1%")
                st.metric("NN F1 (Macro)", "0.61", "↑ 0.02")
            else:
                st.metric("NN Accuracy", "N/A")
                st.metric("NN F1 (Macro)", "N/A")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "This app predicts the probability of loan default using "
            "machine learning models trained on Lending Club data "
            "(~395K loans). The models use 69 features including "
            "loan details, borrower credit history, and categorical variables."
        )

        st.markdown("---")
        st.caption("Built with Streamlit • XGBoost • TensorFlow/Keras")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Analytics", "📋 Batch Processing"])

    with tab1:
        prediction_tab()

    with tab2:
        analytics_tab()

    with tab3:
        batch_tab()

def prediction_tab():
    """Main prediction interface"""

    # Two-column layout
    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💰 Loan Details</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            loan_amnt = st.number_input(
                "Loan Amount ($)",
                min_value=500.0, max_value=40000.0,
                value=DEFAULT_VALUES['loan_amnt'], step=500.0,
                help=FEATURE_DESCRIPTIONS['loan_amnt']
            )
            term = st.selectbox(
                "Term (months)",
                options=[36, 60],
                index=0,
                help=FEATURE_DESCRIPTIONS['term']
            )
            int_rate = st.number_input(
                "Interest Rate (%)",
                min_value=5.0, max_value=30.0,
                value=DEFAULT_VALUES['int_rate'], step=0.1,
                help=FEATURE_DESCRIPTIONS['int_rate']
            )
            installment = st.number_input(
                "Monthly Installment ($)",
                min_value=50.0, max_value=1500.0,
                value=DEFAULT_VALUES['installment'], step=10.0,
                help=FEATURE_DESCRIPTIONS['installment']
            )
        with col2:
            annual_inc = st.number_input(
                "Annual Income ($)",
                min_value=10000.0, max_value=500000.0,
                value=DEFAULT_VALUES['annual_inc'], step=1000.0,
                help=FEATURE_DESCRIPTIONS['annual_inc']
            )
            dti = st.number_input(
                "Debt-to-Income Ratio (%)",
                min_value=0.0, max_value=50.0,
                value=DEFAULT_VALUES['dti'], step=0.5,
                help=FEATURE_DESCRIPTIONS['dti']
            )
            earliest_cr_line = st.number_input(
                "Earliest Credit Line (Year)",
                min_value=1950, max_value=2025,
                value=DEFAULT_VALUES['earliest_cr_line'],
                help=FEATURE_DESCRIPTIONS['earliest_cr_line']
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # Credit History
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Credit History</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            open_acc = st.number_input(
                "Open Accounts",
                min_value=0, max_value=50,
                value=DEFAULT_VALUES['open_acc'],
                help=FEATURE_DESCRIPTIONS['open_acc']
            )
            pub_rec = st.number_input(
                "Public Records",
                min_value=0, max_value=10,
                value=DEFAULT_VALUES['pub_rec'],
                help=FEATURE_DESCRIPTIONS['pub_rec']
            )
        with col2:
            revol_bal = st.number_input(
                "Revolving Balance ($)",
                min_value=0.0, max_value=500000.0,
                value=DEFAULT_VALUES['revol_bal'], step=1000.0,
                help=FEATURE_DESCRIPTIONS['revol_bal']
            )
            revol_util = st.number_input(
                "Revolving Utilization (%)",
                min_value=0.0, max_value=150.0,
                value=DEFAULT_VALUES['revol_util'], step=1.0,
                help=FEATURE_DESCRIPTIONS['revol_util']
            )
        with col3:
            total_acc = st.number_input(
                "Total Accounts",
                min_value=0, max_value=100,
                value=DEFAULT_VALUES['total_acc'],
                help=FEATURE_DESCRIPTIONS['total_acc']
            )
            mort_acc = st.number_input(
                "Mortgage Accounts",
                min_value=0, max_value=20,
                value=DEFAULT_VALUES['mort_acc'],
                help=FEATURE_DESCRIPTIONS['mort_acc']
            )

        pub_rec_bankruptcies = st.number_input(
            "Public Record Bankruptcies",
            min_value=0, max_value=5,
            value=DEFAULT_VALUES['pub_rec_bankruptcies'],
            help=FEATURE_DESCRIPTIONS['pub_rec_bankruptcies']
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # Categorical features
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏷️ Loan & Borrower Characteristics</div>', unsafe_allow_html=True)

        cat_col1, cat_col2 = st.columns(2)

        with cat_col1:
            # Sub-grade
            sub_grade_options = ['A1', 'A2', 'A3', 'A4', 'A5',
                                'B1', 'B2', 'B3', 'B4', 'B5',
                                'C1', 'C2', 'C3', 'C4', 'C5',
                                'D1', 'D2', 'D3', 'D4', 'D5',
                                'E1', 'E2', 'E3', 'E4', 'E5',
                                'F1', 'F2', 'F3', 'F4', 'F5',
                                'G1', 'G2', 'G3', 'G4', 'G5']
            sub_grade = st.selectbox(
                "Loan Sub-Grade",
                options=sub_grade_options,
                index=sub_grade_options.index('B4'),
                help="LC assigned loan grade (A=best, G=worst)"
            )

            # Home ownership
            home_ownership = st.selectbox(
                "Home Ownership",
                options=['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
                index=0,
                help="Borrower's home ownership status"
            )

            # Verification status
            verification_status = st.selectbox(
                "Income Verification",
                options=['Not Verified', 'Source Verified', 'Verified'],
                index=0,
                help="Income verification status"
            )

        with cat_col2:
            # Purpose
            purpose_options = [
                'debt_consolidation', 'credit_card', 'home_improvement',
                'major_purchase', 'small_business', 'medical',
                'car', 'moving', 'house', 'wedding',
                'vacation', 'educational', 'renewable_energy',
                'other'
            ]
            purpose = st.selectbox(
                "Loan Purpose",
                options=purpose_options,
                index=0,
                help="Purpose of the loan"
            )

            # Application type
            application_type = st.selectbox(
                "Application Type",
                options=['INDIVIDUAL', 'JOINT'],
                index=0,
                help="Individual or joint application"
            )

            # Initial list status
            initial_list_status = st.selectbox(
                "Initial List Status",
                options=['w', 'f'],
                index=0,
                help="w=Whole loan, f=Fractional loan"
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # Predict button
        predict_btn = st.button(
            "🔮 Predict Default Risk",
            type="primary",
            use_container_width=True
        )

    with right_col:
        if predict_btn:
            # Prepare input
            inputs = {
                'loan_amnt': loan_amnt,
                'term': term,
                'int_rate': int_rate,
                'installment': installment,
                'annual_inc': annual_inc,
                'dti': dti,
                'earliest_cr_line': earliest_cr_line,
                'open_acc': open_acc,
                'pub_rec': pub_rec,
                'revol_bal': revol_bal,
                'revol_util': revol_util,
                'total_acc': total_acc,
                'mort_acc': mort_acc,
                'pub_rec_bankruptcies': pub_rec_bankruptcies,
            }

            # Add categorical features
            # Sub-grade (one-hot)
            for sg in SUB_GRADE_FEATURES:
                inputs[sg] = 1 if sg == sub_grade else 0

            # Home ownership
            for ho in HOME_OWNERSHIP_FEATURES:
                inputs[ho] = 1 if ho == home_ownership else 0

            # Verification status
            vs_map = {'Not Verified': 'verification_status_Not Verified',
                     'Source Verified': 'verification_status_Source Verified',
                     'Verified': 'verification_status_Verified'}
            for vs in VERIFICATION_FEATURES:
                inputs[vs] = 1 if vs == vs_map[verification_status] else 0

            # Application type
            for at in APPLICATION_FEATURES:
                inputs[at] = 1 if at == f'application_type_{application_type}' else 0

            # Initial list status
            for il in INITIAL_LIST_FEATURES:
                inputs[il] = 1 if il == f'initial_list_status_{initial_list_status}' else 0

            # Purpose
            for p in PURPOSE_FEATURES:
                inputs[p] = 1 if p == f'purpose_{purpose}' else 0

            # Prepare and predict
            input_df = prepare_input_data(inputs)

            with st.spinner("Running prediction..."):
                prob_xgb = predict_xgb(input_df)
                prob_keras = predict_keras(input_df) if 'Ensemble' in selected_model or 'Neural Network' in selected_model else None

                # Use ensemble or selected model
                if 'Ensemble' in selected_model and prob_keras is not None:
                    prob = (prob_xgb + prob_keras) / 2
                elif 'Neural Network' in selected_model and prob_keras is not None:
                    prob = prob_keras
                else:
                    prob = prob_xgb

            # Get risk level
            risk_level, risk_label, risk_emoji = get_risk_level(prob)

            # Display results
            st.markdown(f"""
            <div class="risk-card risk-{risk_level}">
                <div class="prob-label">Probability of Full Repayment</div>
                <div class="prob-text" style="color: {'#16a34a' if risk_level=='low' else '#f59e0b' if risk_level=='medium' else '#dc2626'};">
                    {prob*100:.1f}%
                </div>
                <div style="text-align: center; font-size: 1.25rem; font-weight: 600; margin-top: 0.5rem;">
                    {risk_emoji} {risk_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            st.plotly_chart(create_gauge_chart(prob), use_container_width=True)

            # Model comparison if applicable
            if prob_keras is not None:
                st.plotly_chart(create_risk_breakdown_chart(prob_xgb, prob_keras), use_container_width=True)

            # Risk interpretation
            st.markdown("### 📋 Risk Assessment")

            if risk_level == 'high':
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ High Default Risk</strong><br>
                    This loan application shows characteristics associated with higher default probability.
                    Consider: higher interest rate, requiring collateral, reducing loan amount, or declining.
                </div>
                """, unsafe_allow_html=True)
            elif risk_level == 'medium':
                st.markdown("""
                <div class="warning-box">
                    <strong>⚡ Medium Default Risk</strong><br>
                    This loan has moderate risk indicators. Standard underwriting with careful review recommended.
                    Consider standard terms with additional verification.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <strong>✅ Low Default Risk</strong><br>
                    This loan application shows strong creditworthiness indicators.
                    Favorable terms and standard processing recommended.
                </div>
                """, unsafe_allow_html=True)

            # Key factors
            st.markdown("### 🎯 Key Risk Factors")

            factors = []
            if int_rate > 15: factors.append(f"🔴 High interest rate ({int_rate}%)")
            elif int_rate > 10: factors.append(f"🟡 Moderate interest rate ({int_rate}%)")
            else: factors.append(f"🟢 Low interest rate ({int_rate}%)")

            if dti > 30: factors.append(f"🔴 High DTI ratio ({dti}%)")
            elif dti > 20: factors.append(f"🟡 Moderate DTI ratio ({dti}%)")
            else: factors.append(f"🟢 Low DTI ratio ({dti}%)")

            if revol_util > 70: factors.append(f"🔴 High credit utilization ({revol_util}%)")
            elif revol_util > 40: factors.append(f"🟡 Moderate credit utilization ({revol_util}%)")
            else: factors.append(f"🟢 Low credit utilization ({revol_util}%)")

            if pub_rec > 0: factors.append(f"🔴 Public records present ({pub_rec})")
            else: factors.append("🟢 No public records")

            if pub_rec_bankruptcies > 0: factors.append(f"🔴 Bankruptcy history ({pub_rec_bankruptcies})")
            else: factors.append("🟢 No bankruptcy history")

            sub_grade_risk = {'A': '🟢', 'B': '🟢', 'C': '🟡', 'D': '🟡', 'E': '🟠', 'F': '🔴', 'G': '🔴'}
            factors.append(f"{sub_grade_risk.get(sub_grade[0], '⚪')} Sub-grade: {sub_grade}")

            for factor in factors:
                st.markdown(f"- {factor}")

            # Feature importance
            st.markdown("### 📊 Feature Importance (Top 15)")
            st.plotly_chart(create_feature_importance_chart(input_df), use_container_width=True)

        else:
            # Empty state
            st.markdown("""
            <div class="risk-card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🔮</div>
                <h3>Ready to Predict</h3>
                <p style="color: #64748b;">Fill in the loan details on the left and click <strong>Predict Default Risk</strong> to see the assessment.</p>
            </div>
            """, unsafe_allow_html=True)

def analytics_tab():
    """Analytics and model insights"""
    st.markdown("### 📈 Model Analytics & Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 XGBoost Feature Importance (Global)")

        # Create global feature importance chart
        importances = xgb_model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).nlargest(20, 'Importance').sort_values('Importance', ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=fi_df['Feature'],
            x=fi_df['Importance'],
            orientation='h',
            marker=dict(
                color=fi_df['Importance'],
                colorscale='Blues',
                showscale=False
            ),
            hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
        ))

        fig.update_layout(
            height=600,
            margin=dict(l=150, r=20, t=40, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter, sans-serif'},
            xaxis_title="Importance Score"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Model Performance Comparison")

        # Performance metrics table
        perf_data = {
            'Metric': ['Accuracy', 'Precision (Class 0)', 'Recall (Class 0)',
                      'Precision (Class 1)', 'Recall (Class 1)', 'Macro F1', 'AUC-ROC'],
            'XGBoost': ['75%', '39%', '48%', '87%', '82%', '0.64', '0.82'],
        }

        if has_keras:
            perf_data['Neural Network'] = ['69%', '34%', '60%', '88%', '71%', '0.61', '0.79']

        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        st.markdown("#### 🎯 Optimal Thresholds")
        st.info(
            "**XGBoost**: Optimal threshold = 0.40 (maximizes macro F1)\n\n"
            "**Neural Network**: Optimal threshold = 0.40 (maximizes macro F1)\n\n"
            "Default threshold of 0.5 favors precision for 'Fully Paid' class. "
            "Lower threshold improves recall for 'Charged Off' detection."
        )

        st.markdown("#### 📋 Confusion Matrix (XGBoost @ threshold=0.40)")

        # Confusion matrix visualization
        cm_data = [[7442, 8063],  # Charged Off: TN, FP
                   [11379, 52160]] # Fully Paid: FN, TP

        fig = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=['Predicted: Charged Off', 'Predicted: Fully Paid'],
            y=['Actual: Charged Off', 'Actual: Fully Paid'],
            colorscale='Blues',
            text=cm_data,
            texttemplate='%{text:,}',
            textfont={"size": 16},
            hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
        ))

        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter, sans-serif'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics breakdown
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("True Positive Rate (Recall)", "48%", "Charged Off detected")
            st.metric("True Negative Rate", "82%", "Fully Paid detected")
        with col_b:
            st.metric("Precision (Charged Off)", "39%", "Of predicted defaults")
            st.metric("Precision (Fully Paid)", "87%", "Of predicted repayments")

    # Model architecture info
    st.markdown("---")
    st.markdown("#### 🏗️ Model Architecture")

    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown("""
        **XGBoost Classifier**
        - **Algorithm**: Gradient Boosted Decision Trees
        - **Estimators**: 200
        - **Max Depth**: 6
        - **Learning Rate**: 0.1
        - **Scale Pos Weight**: Balanced for class imbalance
        - **Objective**: Binary logistic regression
        """)

    with arch_col2:
        if has_keras:
            st.markdown("""
            **Neural Network (Keras)**
            - **Architecture**: 4-layer MLP with BatchNorm & Dropout
            - **Layers**: 256 → 128 → 64 → 32 → 1
            - **Activations**: ReLU (hidden), Sigmoid (output)
            - **Regularization**: BatchNorm + Dropout (0.3, 0.3, 0.2)
            - **Optimizer**: Adam
            - **Loss**: Binary Crossentropy
            - **Class Weights**: Balanced (2.55 : 0.62)
            """)
        else:
            st.info("Neural Network model not available in this deployment.")

def batch_tab():
    """Batch prediction from CSV upload"""
    st.markdown("### 📋 Batch Prediction")
    st.markdown("Upload a CSV file with loan applications to get predictions for multiple loans at once.")

    # Required columns info
    with st.expander("📋 Required CSV Format", expanded=False):
        st.markdown("""
        Your CSV should contain the following columns (exact names):

        **Numerical Features:**
        - `loan_amnt`, `term`, `int_rate`, `installment`, `annual_inc`, `dti`
        - `earliest_cr_line`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`
        - `total_acc`, `mort_acc`, `pub_rec_bankruptcies`

        **Categorical Features:**
        - `sub_grade` (e.g., 'B4', 'C2', 'A1')
        - `home_ownership` ('RENT', 'MORTGAGE', 'OWN', 'OTHER')
        - `verification_status` ('Not Verified', 'Source Verified', 'Verified')
        - `application_type` ('INDIVIDUAL', 'JOINT')
        - `initial_list_status` ('w', 'f')
        - `purpose` ('debt_consolidation', 'credit_card', 'home_improvement', etc.)

        **Optional:**
        - `id` or `loan_id` - for identifying rows in output
        """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV with the required columns"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.markdown("#### 📄 Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

            # Validate columns
            required_numerical = ['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti',
                                 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
                                 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']

            missing_num = [c for c in required_numerical if c not in df.columns]
            missing_cat = []

            if 'sub_grade' not in df.columns: missing_cat.append('sub_grade')
            if 'home_ownership' not in df.columns: missing_cat.append('home_ownership')
            if 'verification_status' not in df.columns: missing_cat.append('verification_status')
            if 'application_type' not in df.columns: missing_cat.append('application_type')
            if 'initial_list_status' not in df.columns: missing_cat.append('initial_list_status')
            if 'purpose' not in df.columns: missing_cat.append('purpose')

            if missing_num or missing_cat:
                st.error(f"Missing required columns:\nNumerical: {missing_num}\nCategorical: {missing_cat}")
            else:
                if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner(f"Processing {len(df)} loans..."):
                        results = []

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, row in df.iterrows():
                            # Prepare input
                            inputs = {col: row[col] for col in required_numerical}

                            # Categorical encoding
                            for sg in SUB_GRADE_FEATURES:
                                inputs[sg] = 1 if sg == row['sub_grade'] else 0

                            for ho in HOME_OWNERSHIP_FEATURES:
                                inputs[ho] = 1 if ho == row['home_ownership'] else 0

                            vs_map = {'Not Verified': 'verification_status_Not Verified',
                                     'Source Verified': 'verification_status_Source Verified',
                                     'Verified': 'verification_status_Verified'}
                            for vs in VERIFICATION_FEATURES:
                                inputs[vs] = 1 if vs == vs_map.get(row['verification_status'], '') else 0

                            for at in APPLICATION_FEATURES:
                                inputs[at] = 1 if at == f'application_type_{row["application_type"]}' else 0

                            for il in INITIAL_LIST_FEATURES:
                                inputs[il] = 1 if il == f'initial_list_status_{row["initial_list_status"]}' else 0

                            for p in PURPOSE_FEATURES:
                                inputs[p] = 1 if p == f'purpose_{row["purpose"]}' else 0

                            input_df = prepare_input_data(inputs)
                            prob = predict_xgb(input_df)
                            risk_level, risk_label, _ = get_risk_level(prob)

                            result = {
                                'loan_id': row.get('id', row.get('loan_id', idx)),
                                'default_probability': prob,
                                'risk_level': risk_level,
                                'risk_label': risk_label,
                                'loan_amnt': row['loan_amnt'],
                                'int_rate': row['int_rate'],
                                'sub_grade': row['sub_grade']
                            }
                            results.append(result)

                            if idx % 100 == 0:
                                progress_bar.progress(min((idx + 1) / len(df), 1.0))
                                status_text.text(f"Processed {idx + 1}/{len(df)} loans...")

                        progress_bar.progress(1.0)
                        status_text.text("Complete!")

                    results_df = pd.DataFrame(results)

                    st.success(f"✅ Processed {len(results_df)} loans successfully!")

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Loans", len(results_df))
                    with col2:
                        high_risk = len(results_df[results_df['risk_level'] == 'high'])
                        st.metric("High Risk", high_risk, f"{high_risk/len(results_df)*100:.1f}%")
                    with col3:
                        med_risk = len(results_df[results_df['risk_level'] == 'medium'])
                        st.metric("Medium Risk", med_risk, f"{med_risk/len(results_df)*100:.1f}%")
                    with col4:
                        low_risk = len(results_df[results_df['risk_level'] == 'low'])
                        st.metric("Low Risk", low_risk, f"{low_risk/len(results_df)*100:.1f}%")

                    # Results table
                    st.markdown("#### 📊 Prediction Results")
                    display_df = results_df[['loan_id', 'loan_amnt', 'int_rate', 'sub_grade',
                                             'default_probability', 'risk_level', 'risk_label']].copy()
                    display_df['default_probability'] = display_df['default_probability'].apply(lambda x: f"{x*100:.1f}%")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results as CSV",
                        csv,
                        file_name="loan_default_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    # Risk distribution chart
                    st.markdown("#### 📈 Risk Distribution")
                    risk_counts = results_df['risk_level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        color=risk_counts.index,
                        color_discrete_map={'high': '#dc2626', 'medium': '#f59e0b', 'low': '#16a34a'},
                        title="Portfolio Risk Distribution"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

if __name__ == "__main__":
    main()