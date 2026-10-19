import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_artifact, predict_transactions
from src.chatbot import get_chat_response

import pandas as pd
import streamlit as st
import numpy as np


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_model.joblib"

st.set_page_config(
    page_title="FraudLens AI | Intelligent Credit Card Fraud Detection",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)
def apply_global_style():
    st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background:
            radial-gradient(
                ellipse at 80% 10%,
                #14285b 0%,
                transparent 45%
            ),
            linear-gradient(135deg, #071124, #0b1530);
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0b1530 0%,
            #101c3c 55%,
            #14285b 100%
        );
        border-right: 1px solid #263d70;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #dce7ff;
    }
    div[data-testid="stDownloadButton"] button {
    background: linear-gradient(90deg, #526dfe, #9a5bfa);
    color: white;
    border: none;
    font-weight: 700;
    min-height: 48px;
    border-radius: 12px;
}

    div[data-testid="stDownloadButton"] button:hover {
        border: 1px solid #72e4ff;
        color: white;
    }

    /* Navigation spacing */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 12px;
    }

    /* Navigation options */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(24, 42, 81, 0.75);
        border: 1px solid #263d70;
        border-radius: 12px;
        padding: 12px 14px;
        transition: all 0.2s ease;
    }

    /* Hover effect */
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(54, 217, 255, 0.12);
        border-color: #36d9ff;
    }

    /* Selected navigation item */
    [data-testid="stSidebar"] [role="radiogroup"] label:has(
        input:checked
    ) {
        background: linear-gradient(
            100deg,
            rgba(54, 217, 255, 0.18),
            rgba(139, 124, 255, 0.22)
        );
        border: 1px solid #36d9ff;
    }

    /* Sidebar branding */
    .sidebar-brand {
        padding: 12px 4px 24px 4px;
        border-bottom: 1px solid #263d70;
        margin-bottom: 22px;
    }

    .sidebar-brand .brand-name {
        font-size: 25px;
        font-weight: 800;
        color: #f4f7ff;
        letter-spacing: -0.7px;
    }

    .sidebar-brand .brand-ai {
        color: #36d9ff;
    }

    .sidebar-brand .brand-caption {
        color: #91a5c7;
        font-size: 12px;
        margin-top: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar branding
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:34px;">🔎</div>
        <div class="brand-name">
            FraudLens <span class="brand-ai">AI</span>
        </div>
        <div class="brand-caption">
            INTELLIGENT FRAUD DETECTION
        </div>
    </div>
    """, unsafe_allow_html=True)

apply_global_style()

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Fraud Detection", "Fraud Analytics", "AI Fraud Assistant"],
    key="page_navigation"
)

def go_to_page(target):
    st.session_state.page_navigation = target


def render_home():
    st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 80% 10%,
            #14285b 0%, transparent 45%),
            linear-gradient(135deg, #071124, #0b1530);
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .hero {
        padding: 45px 10px 35px 10px;
    }

    .eyebrow {
        display: inline-block;
        padding: 9px 16px;
        border-radius: 30px;
        background: #172c60;
        color: #8bdcff;
        border: 1px solid #344c90;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: clamp(36px, 5vw, 58px);
        font-weight: 800;
        line-height: 1.12;
        color: #f4f7ff;
        letter-spacing: -1.5px;
    }

    .tagline {
        font-size: 0.68em;
        line-height: 1.2;
        letter-spacing: -0.8px;
    }

    .gradient-text {
        background: linear-gradient(
            90deg, #36d9ff, #8b7cff, #d66cff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        color: #bdcbe6;
        font-size: 17px;
        line-height: 1.8;
        margin-top: 22px;
    }

    .feature-card {
        background: linear-gradient(
            145deg, rgba(24, 42, 81, .85),
            rgba(10, 20, 43, .9)
        );
        border: 1px solid #263d70;
        border-radius: 16px;
        padding: 24px 20px;
        min-height: 205px;
        margin-bottom: 14px;
    }

    .feature-icon {
        font-size: 30px;
        margin-bottom: 12px;
    }

    .feature-title {
        color: #f5f7ff;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .feature-description {
        color: #b7c5df;
        line-height: 1.7;
        font-size: 14px;
    }

    .section-heading {
        color: #f4f7ff;
        font-size: 30px;
        font-weight: 750;
        text-align: center;
        margin: 30px 0 8px 0;
    }

    .section-description {
        color: #b7c5df;
        text-align: center;
        margin-bottom: 30px;
    }

    .trust-note {
        border-top: 1px solid #26375b;
        margin-top: 35px;
        padding-top: 20px;
        color: #91a5c7;
        text-align: center;
        font-size: 13px;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #536dfe, #a45bfa);
        border: none;
        border-radius: 12px;
        color: white;
        min-height: 48px;
        font-weight: 650;
    }

    div.stButton > button[kind="secondary"] {
        border: 1px solid #607bca;
        border-radius: 12px;
        min-height: 48px;
    }

    /* ===== FraudLens AI Sidebar Theme ===== */

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0b1530 0%,
            #101c3c 55%,
            #14285b 100%
        );
        border-right: 1px solid #263d70;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #dce7ff;
    }

    /* Navigation radio options */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 12px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(24, 42, 81, 0.75);
        border: 1px solid #263d70;
        border-radius: 12px;
        padding: 12px 14px;
        transition: all 0.2s ease;
    }

    /* Hover effect */
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(54, 217, 255, 0.12);
        border-color: #36d9ff;
    }

    /* Selected navigation option */
    [data-testid="stSidebar"] [role="radiogroup"] label:has(
        input:checked
    ) {
        background: linear-gradient(
            100deg,
            rgba(54, 217, 255, 0.18),
            rgba(139, 124, 255, 0.22)
        );
        border: 1px solid #36d9ff;
    }

    /* Sidebar radio indicator */
    [data-testid="stSidebar"] [role="radio"] {
        accent-color: #36d9ff;
    }
    </style>
    """, unsafe_allow_html=True)

    # HERO SECTION
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("""
        <div class="hero-title">
            FraudLens AI<br>
            <span class="gradient-text tagline">
                See the Risk Before It Strikes.
            </span>
        </div>

        <div class="hero-description">
            An intelligent credit card fraud detection platform
            powered by Machine Learning and Generative AI.
            Analyze individual transactions, estimate fraud risk,
            process transaction datasets in bulk, and get
            instant answers from your AI Fraud Assistant.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.button(
                "🚀 Get Started",
                type="primary",
                use_container_width=True,
                on_click=go_to_page,
                args=("Fraud Detection",),
                key="home_get_started",
            )

        with c2:
            st.button(
                "🤖 Ask AI Assistant",
                use_container_width=True,
                on_click=go_to_page,
                args=("AI Fraud Assistant",),
                key="home_ai_assistant",
            )

    with right:
        st.markdown("""
        <div style="
            min-height:340px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            text-align:center;
            background:radial-gradient(
                circle, #153e78 0%, #101c3c 48%,
                transparent 72%
            );
            border-radius:30px;
        ">
            <div style="font-size:105px; line-height:1.4;">
                🔎
            </div>
            <div style="
                font-size:25px;
                font-weight:750;
                color:#57e7ff;
            ">
                See Risk Before It Strikes
            </div>
            <div style="
                color:#c1d2ee;
                margin-top:12px;
                line-height:1.8;
            ">
                Smart predictions. Risk insights.<br>
                AI-powered assistance.
            </div>
            <div style="
                margin-top:22px;
                padding:10px 18px;
                border:1px solid #286a81;
                border-radius:30px;
                color:#65f3d0;
                background:#102b3a;
            ">
                ● ML + AI Integration
            </div>
        </div>
        """, unsafe_allow_html=True)

    # FEATURES SECTION
    st.markdown("""
    <div class="section-heading">
        Everything You Need to Detect Fraud
    </div>
    <div class="section-description">
        One platform combining predictive machine learning
        and conversational AI.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    features = [
        (
            "🧠",
            "Machine Learning",
            "Analyze transaction features and estimate the "
            "probability of potential credit card fraud."
        ),
        (
            "📊",
            "Batch Analysis",
            "Upload a transaction CSV, review predictions, "
            "and download the results."
        ),
        (
            "🤖",
            "FraudLens AI Assistant",
            "Ask questions about fraud detection, model "
            "metrics, thresholds, and machine learning."
        ),
    ]

    for col, (icon, title, description) in zip(
        [col1, col2, col3], features
    ):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-description">
                    {description}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # HOW IT WORKS
    st.markdown("""
    <div class="section-heading">
        How It Works
    </div>
    """, unsafe_allow_html=True)

    a, b, c = st.columns(3)

    a.info("**01 · Input**\n\nEnter transaction details or upload a CSV.")
    b.info("**02 · Analyze**\n\nThe trained model evaluates transaction features.")
    c.info("**03 · Review**\n\nReview estimated risk and threshold-based predictions.")

    st.markdown("""
    <div class="trust-note">
        FraudLens AI · AI-Powered Credit Card Fraud Detection<br>
    </div>
    """, unsafe_allow_html=True)
# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

def render_fraud_detection():

    st.markdown("""
    <style>
    /* FraudLens AI - Fraud Detection page */

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    .fraud-header {
        padding: 28px 30px;
        margin-bottom: 25px;
        border-radius: 22px;
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(55, 217, 255, .15),
                transparent 40%
            ),
            linear-gradient(135deg, #111e3c, #10162c);
        border: 1px solid #293e6b;
    }

    .fraud-brand {
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #62e5ff;
        margin-bottom: 15px;
    }

    .fraud-heading {
        font-size: clamp(28px, 4vw, 43px);
        font-weight: 800;
        line-height: 1.2;
        color: #f4f7ff;
        letter-spacing: -1px;
    }

    .fraud-gradient {
        background: linear-gradient(
            90deg, #42dcff, #a78bfa
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .fraud-subtitle {
        color: #b6c7e6;
        font-size: 15px;
        line-height: 1.8;
        margin-top: 12px;
    }

    .fraud-tag {
        display: inline-block;
        margin-top: 18px;
        padding: 7px 13px;
        border-radius: 30px;
        background: #142c3c;
        border: 1px solid #276176;
        color: #72efd8;
        font-size: 12px;
        font-weight: 600;
    }

    /* Form and input styling */
    div[data-testid="stForm"] {
        background: rgba(17, 29, 57, .65);
        border: 1px solid #293e6b;
        border-radius: 18px;
        padding: 22px;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #152342;
        color: #f3f7ff;
        border-radius: 10px;
    }

    div[data-testid="stNumberInput"] label {
        color: #dce7ff;
        font-weight: 600;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(19, 34, 65, .65);
        border: 1px dashed #526fa9;
        border-radius: 14px;
        padding: 12px;
    }

    div[data-testid="stButton"] button,
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 11px;
        min-height: 45px;
        font-weight: 650;
        transition: all .2s ease;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #526dfe, #9a5bfa);
        color: white;
        border: none;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        border: 1px solid #71e8ff;
        color: white;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg, #17294c, #111a32
        );
        border: 1px solid #2b4676;
        padding: 18px;
        border-radius: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #b8c9e8;
    }

    div[data-testid="stMetricValue"] {
        color: #68e4ff;
    }

    div[data-testid="stProgressBar"] > div {
        background-color: #243758;
        border-radius: 20px;
    }

    div[data-testid="stProgressBar"] div[role="progressbar"] {
        background: linear-gradient(90deg, #42dfff, #9c72ff);
        border-radius: 20px;
    }

    /* Keep headings consistent with the Home page */
    h2, h3 {
        color: #edf3ff;
    }

    hr {
        border-color: #293e6b;
    }
    </style>

    <div class="fraud-header">
        <div class="fraud-brand">🔵 FraudLens AI</div>
        <div class="fraud-heading">
            Intelligent Credit Card<br>
            <span class="fraud-gradient">
                Fraud Detection
            </span>
        </div>
        <div class="fraud-subtitle">
            Analyze transactions, estimate fraud risk, and review
            model predictions through one intelligent dashboard.
        </div>
        <div class="fraud-tag">
            ✦ MACHINE LEARNING &nbsp; | &nbsp; BINARY CLASSIFICATION
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "Evaluate individual transactions or analyze an entire "
        "CSV file using your trained fraud detection model."
    )

    st.write(
        "Analyze a transaction using the trained machine learning model "
        "and estimate its probability of being fraudulent."
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.info(
            "Save the trained model in the project-level models folder "
            "before running this application."
        )
        st.stop()


    @st.cache_resource
    def get_artifact():
        return load_artifact()

    try:
        artifact = get_artifact()
        model = artifact["model"]
        threshold = float(artifact["threshold"])
        feature_names = artifact["features"]
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()


    expected_features = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
    ]

    if feature_names != expected_features:
        st.error(
            "The saved model's feature names or order do not match "
            "the expected dataset schema."
        )
        st.stop()

    st.markdown("""
    <div style="
        padding: 13px 18px;
        margin: 15px 0 25px 0;
        background: rgba(30, 180, 140, .09);
        border: 1px solid #246d62;
        border-radius: 12px;
        color: #79f0cc;
        font-weight: 600;
    ">
        ● &nbsp; ML Model Loaded Successfully
        <span style="float:right; color:#9aaccb; font-weight:400;">
            FraudLens AI
        </span>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # 3. TRANSACTION INPUT
    # --------------------------------------------------

    st.subheader("Transaction details")

    st.warning(
        "This dataset uses anonymized PCA features V1–V28. "
        "For a meaningful prediction, use values from a compatible "
        "transaction record. The default values below are placeholders."
    )

    with st.form("fraud_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            time_value = st.number_input(
                "Time (seconds since first transaction)",
                min_value=0.0,
                value=10000.0,
                step=100.0,
            )

        with col2:
            amount_value = st.number_input(
                "Transaction amount",
                min_value=0.0,
                value=100.0,
                step=1.0,
            )

        st.markdown("#### 🧠 Transaction Intelligence")
        st.caption(
            "Enter the anonymized PCA features (V1–V28) for the transaction. "
        )

        feature_values = {}
        columns = st.columns(4)

        for i in range(1, 29):
            feature_name = f"V{i}"

            with columns[(i - 1) % 4]:
                feature_values[feature_name] = st.number_input(
                    feature_name,
                    value=0.0,
                    format="%.6f",
                    key=f"input_{feature_name}",
                )

        submitted = st.form_submit_button(
            "Analyze transaction",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------
    # 4. PREDICTION
    # --------------------------------------------------

    if submitted:

        transaction = {
            "Time": time_value,
            **feature_values,
            "Amount": amount_value,
        }

        input_df = pd.DataFrame(
            [transaction],
            columns=feature_names,
        )

        try:
            fraud_probability = float(
                model.predict_proba(input_df)[0, 1]
            )

            prediction = int(fraud_probability >= threshold)

            st.divider()

            st.markdown("""
            <div style="
                font-size:28px;
                font-weight:800;
                color:#f4f7ff;
                margin-bottom:5px;
            ">
                📊 Analysis Results
            </div>
            <div style="color:#aebfdf; margin-bottom:22px;">
                Your transaction has been evaluated by the trained ML model.
            </div>
            """, unsafe_allow_html=True)

            # Probability and decision threshold
            result_col1, result_col2 = st.columns(2, gap="medium")

            with result_col1:
                st.metric(
                    label="Estimated Fraud Probability",
                    value=f"{fraud_probability:.2%}",
                )

            with result_col2:
                st.metric(
                    label="Decision Threshold",
                    value=f"{threshold:.2%}",
                )

            st.markdown("### Risk Probability")

            st.progress(
                min(max(fraud_probability, 0.0), 1.0),
                text=f"Estimated probability: {fraud_probability:.2%}",
            )

            # Result status card
            if prediction == 1:
                st.markdown(f"""
                <div style="
                    padding:24px;
                    margin-top:18px;
                    border-radius:18px;
                    background:linear-gradient(135deg,#351c35,#25182d);
                    border:1px solid #a94e78;
                ">
                    <div style="font-size:14px;color:#ff9db8;font-weight:700;">
                        ⚠️ TRANSACTION FLAGGED
                    </div>
                    <div style="font-size:25px;font-weight:800;color:#fff0f5;
                                margin-top:10px;">
                        Potential Fraud Detected
                    </div>
                    <div style="color:#e3c5d5;line-height:1.8;margin-top:9px;">
                        The estimated probability meets or exceeds the configured
                        decision threshold. Further verification is recommended.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div style="
                    padding:24px;
                    margin-top:18px;
                    border-radius:18px;
                    background:linear-gradient(135deg,#12372f,#142a35);
                    border:1px solid #287b68;
                ">
                    <div style="font-size:14px;color:#79edc7;font-weight:700;">
                        ✓ BELOW DECISION THRESHOLD
                    </div>
                    <div style="font-size:25px;font-weight:800;color:#effff9;
                                margin-top:10px;">
                        Transaction Not Flagged
                    </div>
                    <div style="color:#c0e2d9;line-height:1.8;margin-top:9px;">
                        The estimated probability is below the configured
                        threshold. This does not guarantee that the transaction
                        is legitimate.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(
                "FraudLens AI provides experimental model estimates. "
                "Results are not a guarantee of fraud or legitimacy."
            )

            st.caption(
                "This is an experimental model, not a production banking "
                "system. A prediction does not guarantee that a transaction "
                "is fraudulent or legitimate."
            )

            if "individual_analytics" not in st.session_state:
                st.session_state["individual_analytics"] = []

            st.session_state["individual_analytics"].append({
                "Fraud_Probability": fraud_probability,
                "Predicted_Class": prediction,
})

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


    # --------------------------------------------------
    # 5. BATCH PREDICTION FROM CSV
    # --------------------------------------------------

    st.divider()

    st.markdown("""
    <div style="
        padding: 22px;
        margin: 10px 0 20px 0;
        border-radius: 16px;
        background: linear-gradient(135deg, #152747, #10182e);
        border: 1px solid #304b79;
    ">
        <div style="font-size:26px; font-weight:750; color:#f4f7ff;">
            📂 Batch Fraud Detection
        </div>
        <div style="margin-top:9px; color:#b8c9e8; line-height:1.7;">
            Analyze multiple transactions at once, review flagged
            records, and export the prediction results.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "Upload a CSV containing multiple transactions to analyze them "
        "together and download the prediction results."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        key="batch_csv_uploader",
    )

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            # The target is optional and is never used as a model input.
            input_columns = [
                column for column in batch_df.columns
                if column != "Class"
            ]

            missing_features = [
                feature for feature in feature_names
                if feature not in input_columns
            ]

            if missing_features:
                st.error(
                    "The CSV is missing required model features: "
                    + ", ".join(missing_features)
                )

            elif batch_df.empty:
                st.error("The uploaded CSV contains no transactions.")

            else:
                # Select features in exactly the order used during training.
                batch_features = batch_df[feature_names].copy()

                # Ensure every feature is numeric.
                for column in feature_names:
                    batch_features[column] = pd.to_numeric(
                        batch_features[column],
                        errors="coerce",
                    )

                # Reject missing or non-finite values.
                import numpy as np

                values = batch_features.to_numpy(dtype=float)

                if not np.isfinite(values).all():
                    st.error(
                        "The CSV contains missing, non-numeric, or "
                        "infinite feature values. Please correct the file."
                    )

                else:
                    # Predict without using the optional Class column.
                    probabilities = model.predict_proba(
                        batch_features
                    )[:, 1]

                    predictions = (
                        probabilities >= threshold
                    ).astype(int)

                    # Preserve original columns and add predictions.
                    results_df = batch_df.copy()

                    results_df["Fraud_Probability"] = probabilities
                    results_df["Predicted_Class"] = predictions
                    results_df["Prediction"] = [
                        "Potential Fraud" if value == 1
                        else "Not Flagged"
                        for value in predictions
                    ]

                    st.success(
                        f"Successfully analyzed {len(results_df):,} "
                        "transactions."
                    )

                    # Summary metrics
                    total = len(results_df)
                    flagged = int(predictions.sum())
                    not_flagged = total - flagged

                    col1, col2, col3 = st.columns(3, gap="medium")

                    with col1:
                        st.markdown(f"""
                        <div style="
                            padding:22px;
                            min-height:125px;
                            border-radius:16px;
                            background:linear-gradient(145deg,#17294c,#111a32);
                            border:1px solid #34517f;
                        ">
                            <div style="color:#b8c9e8;font-size:14px;">
                                📊 Total Transactions
                            </div>
                            <div style="font-size:32px;font-weight:800;
                                        color:#72e4ff;margin-top:12px;">
                                {total:,}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div style="
                            padding:22px;
                            min-height:125px;
                            border-radius:16px;
                            background:linear-gradient(145deg,#3b2035,#24182e);
                            border:1px solid #854667;
                        ">
                            <div style="color:#f0bfd2;font-size:14px;">
                                ⚠️ Potential Fraud
                            </div>
                            <div style="font-size:32px;font-weight:800;
                                        color:#ff9db8;margin-top:12px;">
                                {flagged:,}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div style="
                            padding:22px;
                            min-height:125px;
                            border-radius:16px;
                            background:linear-gradient(145deg,#153a35,#142630);
                            border:1px solid #2b7968;
                        ">
                            <div style="color:#b5e7d9;font-size:14px;">
                                ✓ Not Flagged
                            </div>
                            <div style="font-size:32px;font-weight:800;
                                        color:#79edc7;margin-top:12px;">
                                {not_flagged:,}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("")
                    st.caption(
                        "Counts are based on the model's configured decision threshold."
                    )

                    st.markdown("### 📈 Transaction Analysis Report")
                    st.caption(
                        "Preview of analyzed transactions. Download the CSV below "
                        "to access the complete results."
                    )
                    st.dataframe(
                        results_df.head(100),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Download predictions as CSV
                    csv_data = results_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download prediction results",
                        data=csv_data,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.session_state["batch_results_df"] = results_df.copy()

        except (ValueError, KeyError, pd.errors.ParserError) as exc:
            st.error(f"Could not process the CSV: {exc}")

def render_fraud_analytics():
    import pandas as pd

    st.markdown("""
    <style>
    .fraud-brand {
        color: #55DFFF;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .fraud-heading {
        font-size: clamp(30px, 4vw, 43px);
        font-weight: 800;
        letter-spacing: -1.3px;
        line-height: 1.2;
        color: #F4F7FF;
        margin-bottom: 12px;
    }

    .fraud-gradient {
        background: linear-gradient(
            90deg, #36D9FF, #8B7CFF, #D66CFF
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .fraud-subtitle {
        color: #B7C7E5;
        font-size: 16px;
        line-height: 1.7;
        margin-bottom: 20px;
    }

    .analytics-section-title {
        color: #F4F7FF;
        font-size: 23px;
        font-weight: 750;
        letter-spacing: -0.4px;
        margin-bottom: 14px;
    }

    div[data-testid="stMetricLabel"] p {
        color: #B7C7E5;
        font-size: 14px;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #F4F7FF;
        font-weight: 750;
    }

    .fraud-brand {
    color: #55DFFF;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

    .fraud-heading {
        font-size: clamp(30px, 3.5vw, 40px);
        font-weight: 800;
        letter-spacing: -1.2px;
        line-height: 1.2;
        color: #F4F7FF;
        margin: 0 0 10px 0;
    }

    .fraud-gradient {
        background: linear-gradient(
            90deg, #55DFFF, #8B7CFF
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .fraud-subtitle {
        color: #B7C7E5;
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 18px;
    }

    .analytics-section-title {
        color: #F4F7FF;
        font-size: 22px;
        font-weight: 750;
        letter-spacing: -0.5px;
        margin-bottom: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

    batch_df = st.session_state.get("batch_analytics_df")
    history = st.session_state.get("individual_analytics", [])

    # Prefer the latest batch when one is available.
    if batch_df is not None and not batch_df.empty:
        analytics_df = batch_df.copy()
        source_label = "Latest batch analysis"
    elif history:
        analytics_df = pd.DataFrame(history)
        source_label = "Individual prediction history"
    else:
        st.info(
            "No analytics data yet. Go to Fraud Detection and "
            "analyze a transaction or upload a CSV first."
        )
        return

    required = {"Fraud_Probability", "Predicted_Class"}
    if not required.issubset(analytics_df.columns):
        st.warning("The available prediction data is incomplete.")
        return

    analytics_df["Fraud_Probability"] = pd.to_numeric(
        analytics_df["Fraud_Probability"], errors="coerce"
    )
    analytics_df["Predicted_Class"] = pd.to_numeric(
        analytics_df["Predicted_Class"], errors="coerce"
    )

    analytics_df = analytics_df.dropna(
        subset=["Fraud_Probability", "Predicted_Class"]
    )

    if analytics_df.empty:
        st.warning("There are no valid prediction records to display.")
        return

    total = len(analytics_df)
    flagged = int((analytics_df["Predicted_Class"] == 1).sum())
    not_flagged = int((analytics_df["Predicted_Class"] == 0).sum())
    fraud_rate = (flagged / total) * 100
    avg_probability = analytics_df["Fraud_Probability"].mean() * 100

    st.markdown("""
    <div class="fraud-brand">🔵 FRAUDLENS AI · ANALYTICS</div>
    <div class="fraud-heading">
        Fraud <span class="fraud-gradient">Analytics Dashboard</span>
    </div>
    <div class="fraud-subtitle">
        Monitor transaction predictions, assess fraud risk,
        and explore your model's estimated fraud probability.
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"Data source: {source_label}")

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Analyzed", f"{total:,}")
    c2.metric("Potential Fraud", f"{flagged:,}")
    c3.metric("Not Flagged", f"{not_flagged:,}")
    c4.metric("Flagged Rate", f"{fraud_rate:.2f}%")

    st.divider()

    # Prediction distribution
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
        '<div class="analytics-section-title">'
        '📊 Prediction Breakdown</div>',
        unsafe_allow_html=True
    )

        breakdown = pd.DataFrame({
            "Prediction": ["Potential Fraud", "Not Flagged"],
            "Transactions": [flagged, not_flagged],
        }).set_index("Prediction")

        st.bar_chart(breakdown, height=300)

    with right:
        st.markdown(
        '<div class="analytics-section-title">'
        '📈 Fraud Probability Distribution</div>',
        unsafe_allow_html=True
        )
        st.metric(
            "Average estimated probability",
            f"{avg_probability:.2f}%",
        )

        # Group probabilities into ranges for a readable chart.
        bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
        labels = [
            "0–10%", "10–20%", "20–30%", "30–40%", "40–50%",
            "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"
        ]

        probabilities = analytics_df["Fraud_Probability"].clip(0, 1)

        distribution = pd.cut(
            probabilities,
            bins=bins,
            labels=labels,
            include_lowest=True,
        ).value_counts(sort=False)

        st.bar_chart(
            distribution.rename("Transactions"),
            height=300,
        )

    st.divider()
    st.markdown(
        '<div class="analytics-section-title">'
        '📂 Transaction Intelligence </div>',
        unsafe_allow_html=True
    )

    display_columns = [
        col for col in [
            "Time", "Amount", "Fraud_Probability",
            "Predicted_Class", "Prediction"
        ]
        if col in analytics_df.columns
    ]

    display_df = analytics_df[display_columns].copy()

    if "Fraud_Probability" in display_df.columns:
        display_df["Fraud_Probability"] = (
            display_df["Fraud_Probability"].map(
                lambda x: f"{x:.2%}"
            )
        )

    if "Predicted_Class" in display_df.columns:
        display_df["Predicted_Class"] = (
            display_df["Predicted_Class"].map({
                1: "Potential Fraud",
                0: "Not Flagged",
            }).fillna("Unknown")
        )

    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Analytics Report",
        data=analytics_df.to_csv(index=False).encode("utf-8"),
        file_name="fraudlens_analytics.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Analytics summarize model predictions, not confirmed fraud cases. "
        "A transaction marked Not Flagged is not guaranteed to be legitimate."
    )


def render_ai_assistant():
    st.title("🤖 FraudLens AI Assistant")
    st.caption(
        "Ask questions about fraud detection "
        "and machine learning."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_messages = []
        st.rerun()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask about fraud detection...",
        key="chat_input"
    )

    if prompt:
        st.session_state.chat_messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = get_chat_response(
                        st.session_state.chat_messages
                    )
                    st.markdown(reply)

                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": reply}
                    )

                except Exception as exc:
                    st.error(f"AI assistant error: {exc}")

if page == "Home":
    render_home()

elif page == "Fraud Detection":
    render_fraud_detection()

elif page == "Fraud Analytics":
    render_fraud_analytics()

elif page == "AI Fraud Assistant":
    render_ai_assistant()