import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction System")
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose Architecture",
    [
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "Decision Tree",
        "MLP",
        "CNN",
        "RNN"
    ]
)

# Sidebar: trained models summary
st.sidebar.markdown("## Trained Models")
st.sidebar.success("SVM : F1 = 0.889")
st.sidebar.info("Random Forest : F1 = 0.871")
st.sidebar.info("MLP : F1 = 0.853")
st.sidebar.info("CNN : F1 = 0.847")
st.sidebar.info("RNN : F1 = 0.787")
st.sidebar.markdown(
    """### Best Architecture🏆
SVM
Accuracy = 88.5%
F1 Score = 88.9%
ROC AUC = 93.4%"""
)

# Define PyTorch model classes (same architecture as notebook)
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNClassifier(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        conv_out_length = input_length - 3 + 1
        pooled_length = conv_out_length // 2
        self.fc1 = nn.Linear(16 * pooled_length, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class RNNClassifier(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = hidden[-1]
        out = self.fc(out)
        return out

# Tabs: Predict and Model Comparison
tabs = st.tabs(["Predict", "Model Comparison"])

with tabs[0]:
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1])
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
        trestbps = st.number_input("Resting Blood Pressure", 50, 300, 120)
        chol = st.number_input("Cholesterol", 50, 700, 200)
        fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
        restecg = st.selectbox("Rest ECG", [0, 1, 2])
    with col2:
        thalach = st.number_input("Max Heart Rate", 50, 250, 150)
        exang = st.selectbox("Exercise Angina", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope", [1, 2, 3])
        ca = st.selectbox("CA", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [3, 6, 7])

    if st.button("Predict"):
        patient = pd.DataFrame([
            {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }
        ])

        # Load preprocessor and classic models
        preprocessor = joblib.load("preprocessor.pkl")
        X = preprocessor.transform(patient)

        # Handle sparse output from OneHotEncoder if applicable
        def to_dense(arr):
            try:
                return arr.toarray()
            except Exception:
                return arr

        if model_name == "Logistic Regression":
            model = joblib.load("lr_model.pkl")
            prob = model.predict_proba(X)[0][1]
        elif model_name == "Random Forest":
            model = joblib.load("rf_model.pkl")
            prob = model.predict_proba(X)[0][1]
        elif model_name == "SVM":
            model = joblib.load("svm_model.pkl")
            prob = model.predict_proba(X)[0][1]
        elif model_name == "Decision Tree":
            model = joblib.load("dt_model.pkl")
            prob = model.predict_proba(X)[0][1]
        elif model_name == "MLP":
            X_dense = to_dense(X)
            model = MLPClassifier(input_dim=X_dense.shape[1])
            model.load_state_dict(torch.load("mlp_model.pth", map_location="cpu"))
            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(torch.tensor(X_dense, dtype=torch.float32))).item()
        elif model_name == "CNN":
            X_dense = to_dense(X)
            model = CNNClassifier(input_length=X_dense.shape[1])
            model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
            model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(X_dense, dtype=torch.float32).unsqueeze(1)
                prob = torch.sigmoid(model(x_tensor)).item()
        elif model_name == "RNN":
            X_dense = to_dense(X)
            model = RNNClassifier(input_length=X_dense.shape[1])
            model.load_state_dict(torch.load("rnn_model.pth", map_location="cpu"))
            model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(X_dense, dtype=torch.float32).unsqueeze(-1)
                prob = torch.sigmoid(model(x_tensor)).item()
        else:
            st.error("Selected model not available.")
            st.stop()

        prediction = int(prob > 0.5)
        if prediction == 1:
            st.error(f"Heart Disease Detected\nProbability = {prob:.2%}")
        else:
            st.success(f"No Heart Disease\nProbability = {(1-prob):.2%}")

        st.metric("Disease Probability", f"{prob:.2%}")

with tabs[1]:
    st.subheader("Model Comparison")
    comp = pd.DataFrame([
        ["SVM", "88.5%", "88.9%"],
        ["Random Forest", "86.8%", "87.1%"],
        ["MLP", "85.2%", "85.2%"],
        ["CNN", "85.2%", "84.7%"],
        ["RNN", "78.6%", "78.6%"],
    ], columns=["Model", "Accuracy", "F1"])
    st.table(comp)
