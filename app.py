import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection Using Machine Learning")

# Upload CSV
uploaded_file = st.file_uploader("Upload your credit card dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    # Preprocess
    df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
    df.drop(['Time'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']

    st.write("Class distribution before SMOTE:")
    st.bar_chart(y.value_counts())

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    st.write("Class distribution after SMOTE:")
    st.bar_chart(pd.Series(y_res).value_counts())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    # Train models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    st.subheader("📈 Model Evaluation")
    model_metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        model_metrics[name] = auc

        st.markdown(f"**{name}**")
        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.text(cm)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{name} - Confusion Matrix")
        st.pyplot(fig)

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.success(f"{name} ROC AUC Score: {auc:.4f}")

    # Show comparison
    st.subheader("📊 ROC AUC Score Comparison")
    scores_df = pd.DataFrame.from_dict(model_metrics, orient='index', columns=["ROC AUC"])
    st.bar_chart(scores_df)

    # Real-time prediction
    st.subheader("🔮 Real-Time Fraud Prediction")

    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    selected_model = models[selected_model_name]

    st.markdown("#### Enter Transaction Details (or modify random values):")

    input_data = {}
    for col in X.columns:
        val = float(np.round(np.random.normal(), 4))
        input_data[col] = st.number_input(f"{col}", value=val)

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        prediction = selected_model.predict(input_df)[0]
        prob = selected_model.predict_proba(input_df)[0][1]

        st.markdown(f"### 🔍 Prediction: {'Fraud' if prediction == 1 else 'Not Fraud'}")
        st.markdown(f"**Fraud Probability:** {prob*100:.2f}%")
else:
    st.info("📂 Upload a dataset to get started.")