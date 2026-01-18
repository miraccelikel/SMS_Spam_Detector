import streamlit as st
import joblib
import pandas as pd
import os
import json
from utils import clean_text, tokenizer_porter

# --- Page Config ---
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .spam-highlight { background-color: #ffe6e6; color: #d9534f; font-weight: bold; padding: 2px 5px; border-radius: 4px; }
    .safe-text { color: #5cb85c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# --- Helper: Highlight Spam Words ---
def highlight_spam_words(text):
    # Common spam trigger words to highlight
    triggers = ["win", "winner", "free", "cash", "urgent", "credit", "offer", "click", "link", "buy", "money", "prize",
                "award", "selected", "update", "verify", "limited", "account"]
    words = text.split()
    result = []
    for word in words:
        # Strip punctuation for matching but keep original word for display
        clean = "".join(filter(str.isalpha, word.lower()))
        if clean in triggers:
            result.append(f"<span class='spam-highlight'>{word}</span>")
        else:
            result.append(word)
    return " ".join(result)


# --- Load Model (Cached) ---
@st.cache_resource
def load_model():
    model_path = 'models/spam_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


pipeline = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 Control Panel")

    # 1. Model Metrics Section
    st.subheader("Model Metrics")
    st.caption("Performance metrics on the test dataset.")

    # Load Dynamic Metrics from JSON
    metrics_path = 'models/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        c2.metric("Spam Recall", f"{metrics['recall'] * 100:.1f}%")
    else:
        st.info("Metrics not found. Please run 'train_model.py' first.")

    st.markdown("---")

    # 2. Confusion Matrix Section (Inside Expander)
    with st.expander("🖼️ View Confusion Matrix", expanded=False):
        matrix_path = "assets/confusion_matrix.png"
        if os.path.exists(matrix_path):
            st.image(matrix_path, caption="Confusion Matrix (Test Set)", use_container_width=True)
            st.caption("""
            **How to read:**
            * **Top-Left:** Correctly identified Spam.
            * **Bottom-Right:** Correctly identified Normal messages.
            * **Errors:** Values outside the diagonal represent mistakes.
            """)
        else:
            st.warning("Plot not found. Train the model first.")

    st.markdown("---")

    # 3. Batch Analysis Section
    st.subheader("📂 Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV (Column: 'message')", type=["csv"])

    if uploaded_file and pipeline:
        try:
            df_up = pd.read_csv(uploaded_file)
            if 'message' in df_up.columns:
                with st.spinner('Analyzing...'):
                    # Predict using the loaded pipeline
                    df_up['prediction'] = pipeline.predict(df_up['message'].apply(clean_text))

                st.success(f"Analyzed {len(df_up)} messages.")
                st.dataframe(df_up[['message', 'prediction']].head(5))

                # Download Button
                csv_data = df_up.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", csv_data, "batch_results.csv", "text/csv")
            else:
                st.error("CSV must contain a 'message' column.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- MAIN PAGE ---
st.title("📩 SMS Spam Detector")
st.markdown("""
This application utilizes **Support Vector Machines (SVM)** and **Natural Language Processing (NLP)** to detect spam messages with high accuracy.
**Key Features:**
* Real-time spam detection.
* Keyword highlighting for explainability.
* Batch processing for multiple messages.
""")

# Check if model is loaded
if pipeline is None:
    st.error("🚨 ERROR: Model file not found! Please run 'train_model.py' to generate the model.")
else:
    # Layout: Input on Left, Result on Right
    col_input, col_result = st.columns([1.5, 1])

    with col_input:
        st.subheader("📝 Live Analysis")
        user_input = st.text_area("Paste the message to analyze:", height=250,
                                  placeholder="Ex: URGENT! You have won a $1000 Walmart gift card...")

        analyze_btn = st.button("🔍 Detect Spam", type="primary", use_container_width=True)

    # --- RESULTS AREA ---
    with col_result:
        st.subheader("Prediction Result")

        if analyze_btn and user_input:
            # Preprocess and Predict
            processed = clean_text(user_input)
            pred = pipeline.predict([processed])[0]
            proba = pipeline.predict_proba([processed]).max()

            if pred == 'spam':
                st.error(f"🚨 **SPAM DETECTED**")
                st.progress(proba, text=f"Confidence Score: {proba * 100:.1f}%")

                # Explainability Section
                st.markdown("### 🧐 Suspicious Triggers:")
                highlighted = highlight_spam_words(user_input)
                st.markdown(f"{highlighted}", unsafe_allow_html=True)
                st.caption("*Highlighted words are common spam indicators found in your message.*")

            else:
                st.success(f"✅ **SAFE MESSAGE (HAM)**")
                st.progress(proba, text=f"Confidence Score: {proba * 100:.1f}%")
                st.markdown("Message content appears clean and safe.")

        elif analyze_btn and not user_input:
            st.warning("Please enter a message to analyze.")
        else:
            st.info("👈 Enter a text on the left to start detection.")