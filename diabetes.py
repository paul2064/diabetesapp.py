import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

# Load dataset with error handling
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/paul2064/diabetesapp.py/main/diabetesdataset.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for HTTP failures
        return pd.read_csv(StringIO(response.text))
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load dataset: {e}")
        return None

# Preprocess data
def preprocess_data(df):
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])  # Drop 'Id' column if it exists
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, X.columns

# Train model fresh every time
def train_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            min_samples_split=5, 
            min_samples_leaf=2, 
            max_features="sqrt", 
            random_state=42, 
            class_weight="balanced"
        )
    else:
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, accuracy, report, X_train

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.markdown("# 🩺 Diabetes Prediction App")
st.sidebar.header("🔢 Input Patient Details")

# Load and preprocess data
data = load_data()
if data is not None:
    X, y, scaler, feature_names = preprocess_data(data)

    # Select Model
    model_type = st.sidebar.selectbox("Select Model", ["Random Forest", "Neural Network (MLP)"])

    # Train model fresh every time
    model, accuracy, report, X_train = train_model(X, y, model_type)

    # Display model performance
    st.subheader("📊 Model Performance")
    st.write(f"**Accuracy:** {accuracy:.2f}")

    # Toggle to show/hide classification report
    if st.checkbox("Show Classification Report"):
        st.json(report)

    # Feature Importance (Only for Random Forest)
    if model_type == "Random Forest":
        st.subheader("🔍 Feature Importance")
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y=feature_importance_df['Feature'], x=feature_importance_df['Importance'], palette="viridis", ax=ax)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title("Feature Importance", fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
        st.pyplot(fig)

    # Sidebar User Input
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, value=100)
    bp = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=500, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

    # Predict Button
    if st.sidebar.button("🔍 Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data.reshape(1, -1))  # Ensure proper reshaping
        prediction = model.predict(input_scaled)
        result = "🛑 Positive" if prediction[0] == 1 else "✅ Negative"
        st.markdown(f"## **Prediction: {result}**")
