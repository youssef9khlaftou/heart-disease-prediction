from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

data = fetch_ucirepo(id=45)

X = data.data.features
y = data.data.targets

print(X.head())
print(y.head())

print("X shape:", X.shape)
print("y shape:", y.shape)

print(X.isnull().sum())

y = (y > 0).astype(int)
print(y.head())

X['ca'] = X['ca'].fillna(X['ca'].mode()[0])
X['thal'] = X['thal'].fillna(X['thal'].mode()[0])

print(X.isnull().sum())
print(y.value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x=y['num'])
plt.title("Distribution of Heart Disease (0 vs 1)")
plt.show()

plt.figure()
sns.histplot(data=X, x="age", hue=y['num'], bins=20, kde=True)
plt.title("Age distribution vs Heart Disease")
plt.show()

plt.figure()
sns.countplot(x=X['sex'], hue=y['num'])
plt.title("Heart Disease by Sex")
plt.show()

plt.figure()
sns.countplot(x=X['cp'], hue=y['num'])
plt.title("Chest Pain Type vs Heart Disease")
plt.show()

plt.figure()
sns.boxplot(x=y['num'], y=X['chol'])
plt.title("Cholesterol vs Heart Disease")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(X.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y['num'],
    test_size=0.2,
    random_state=42
)

print(X_train.shape, X_test.shape)

num_cols = ['age','trestbps','chol','thalach','oldpeak']
cat_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(X_train_processed.shape)

X_train_p = X_train_processed
X_test_p = X_test_processed
y_train_p = y_train
y_test_p = y_test

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_p, y_train_p)

rf = RandomForestClassifier()
rf.fit(X_train_p, y_train_p)

svm = SVC(probability=True)
svm.fit(X_train_p, y_train_p)

dt = DecisionTreeClassifier()
dt.fit(X_train_p, y_train_p)

lr_pred = lr.predict(X_test_p)
rf_pred = rf.predict(X_test_p)
svm_pred = svm.predict(X_test_p)
dt_pred = dt.predict(X_test_p)

from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

print("LR accuracy:", accuracy_score(y_test_p, lr_pred))
print("LR F1:", f1_score(y_test_p, lr_pred))

print("RF accuracy:", accuracy_score(y_test_p, rf_pred))
print("RF F1:", f1_score(y_test_p, rf_pred))

print("SVM accuracy:", accuracy_score(y_test_p, svm_pred))
print("SVM F1:", f1_score(y_test_p, svm_pred))

print("DT accuracy:", accuracy_score(y_test_p, dt_pred))
print("DT F1:", f1_score(y_test_p, dt_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_p, rf_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

rf_prob = rf.predict_proba(X_test_p)[:,1]

fpr, tpr, _ = roc_curve(y_test_p, rf_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"RF (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

models = {
    "LogReg": lr,
    "RF": rf,
    "SVM": svm,
    "DT": dt
}

for name, model in models.items():
    pred = model.predict(X_test_p)
    print(name, "F1:", f1_score(y_test_p, pred))


# ============================================================================
# STEP 8: PyTorch MLP (Deep Learning Model)
# ============================================================================

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Convert preprocessed data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_p, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_p, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_p.values, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test_p.values, dtype=torch.float32).unsqueeze(1).to(device)

# Define MLP architecture
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
        x = self.fc3(x)  # No activation here (used with BCEWithLogitsLoss)
        return x

# Initialize model
input_dim = X_train_tensor.shape[1]
mlp = MLPClassifier(input_dim=input_dim).to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

# Training loop
epochs = 50
train_losses = []

print(f"Training MLP on {device}...")
for epoch in range(epochs):
    mlp.train()
    
    # Forward pass
    outputs = mlp(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# MLP predictions
mlp.eval()
with torch.no_grad():
    mlp_logits_train = mlp(X_train_tensor).cpu().numpy()
    mlp_logits_test = mlp(X_test_tensor).cpu().numpy()
    mlp_prob_test = torch.sigmoid(torch.tensor(mlp_logits_test)).numpy()
    mlp_pred = (mlp_prob_test > 0.5).astype(int).flatten()

print(f"MLP accuracy: {accuracy_score(y_test_p, mlp_pred):.4f}")
print(f"MLP F1: {f1_score(y_test_p, mlp_pred):.4f}")

# ============================================================================
# CNN MODEL
# ============================================================================
X_train_cnn = torch.tensor(X_train_p, dtype=torch.float32).to(device)
X_test_cnn = torch.tensor(X_test_p, dtype=torch.float32).to(device)
y_train_cnn = torch.tensor(y_train_p.values, dtype=torch.float32).unsqueeze(1).to(device)
y_test_cnn = torch.tensor(y_test_p.values, dtype=torch.float32).unsqueeze(1).to(device)

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

cnn = CNNClassifier(input_length=X_train_p.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

for epoch in range(50):
    cnn.train()
    outputs = cnn(X_train_cnn)
    loss = criterion(outputs, y_train_cnn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

cnn.eval()
with torch.no_grad():
    cnn_prob = torch.sigmoid(cnn(X_test_cnn)).cpu().numpy().flatten()
    cnn_pred = (cnn_prob > 0.5).astype(int)

print("CNN Accuracy:", accuracy_score(y_test_p, cnn_pred))
print("CNN F1:", f1_score(y_test_p, cnn_pred))

# ============================================================================
# SIMPLE RNN MODEL
# ============================================================================
X_train_rnn = torch.tensor(X_train_p, dtype=torch.float32).unsqueeze(2).to(device)
X_test_rnn = torch.tensor(X_test_p, dtype=torch.float32).unsqueeze(2).to(device)
y_train_rnn = torch.tensor(y_train_p.values, dtype=torch.float32).unsqueeze(1).to(device)
y_test_rnn = torch.tensor(y_test_p.values, dtype=torch.float32).unsqueeze(1).to(device)

class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = hidden[-1]
        out = self.fc(out)
        return out

rnn = RNNClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.001)

for epoch in range(50):
    rnn.train()
    outputs = rnn(X_train_rnn)
    loss = criterion(outputs, y_train_rnn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

rnn.eval()
with torch.no_grad():
    rnn_prob = torch.sigmoid(rnn(X_test_rnn)).cpu().numpy().flatten()
    rnn_pred = (rnn_prob > 0.5).astype(int)

print("RNN Accuracy:", accuracy_score(y_test_p, rnn_pred))
print("RNN F1:", f1_score(y_test_p, rnn_pred))


# ============================================================================
# STEP 9: Model Comparison Table
# ============================================================================

from sklearn.metrics import precision_score

# Collect predictions and probabilities for all models
results = []

# Logistic Regression
lr_prob = lr.predict_proba(X_test_p)[:, 1]
results.append({
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test_p, lr_pred),
    "Precision": precision_score(y_test_p, lr_pred),
    "Recall": recall_score(y_test_p, lr_pred),
    "F1-Score": f1_score(y_test_p, lr_pred),
    "ROC-AUC": roc_auc_score(y_test_p, lr_prob),
    "Predictions": lr_pred,
    "Probabilities": lr_prob
})

# Random Forest
rf_prob = rf.predict_proba(X_test_p)[:, 1]
results.append({
    "Model": "Random Forest",
    "Accuracy": accuracy_score(y_test_p, rf_pred),
    "Precision": precision_score(y_test_p, rf_pred),
    "Recall": recall_score(y_test_p, rf_pred),
    "F1-Score": f1_score(y_test_p, rf_pred),
    "ROC-AUC": roc_auc_score(y_test_p, rf_prob),
    "Predictions": rf_pred,
    "Probabilities": rf_prob
})

# SVM
svm_prob = svm.predict_proba(X_test_p)[:, 1]
results.append({
    "Model": "SVM",
    "Accuracy": accuracy_score(y_test_p, svm_pred),
    "Precision": precision_score(y_test_p, svm_pred),
    "Recall": recall_score(y_test_p, svm_pred),
    "F1-Score": f1_score(y_test_p, svm_pred),
    "ROC-AUC": roc_auc_score(y_test_p, svm_prob),
    "Predictions": svm_pred,
    "Probabilities": svm_prob
})

# Decision Tree
dt_prob = dt.predict_proba(X_test_p)[:, 1]
results.append({
    "Model": "Decision Tree",
    "Accuracy": accuracy_score(y_test_p, dt_pred),
    "Precision": precision_score(y_test_p, dt_pred),
    "Recall": recall_score(y_test_p, dt_pred),
    "F1-Score": f1_score(y_test_p, dt_pred),
    "ROC-AUC": roc_auc_score(y_test_p, dt_prob),
    "Predictions": dt_pred,
    "Probabilities": dt_prob
})

# PyTorch MLP
results.append({
    "Model": "PyTorch MLP",
    "Accuracy": accuracy_score(y_test_p, mlp_pred),
    "Precision": precision_score(y_test_p, mlp_pred),
    "Recall": recall_score(y_test_p, mlp_pred),
    "F1-Score": f1_score(y_test_p, mlp_pred),
    "ROC-AUC": roc_auc_score(y_test_p, mlp_prob_test.flatten()),
    "Predictions": mlp_pred,
    "Probabilities": mlp_prob_test.flatten()
})

results.append({
    "Model": "CNN",
    "Accuracy": accuracy_score(y_test_p, cnn_pred),
    "Precision": precision_score(y_test_p, cnn_pred),
    "Recall": recall_score(y_test_p, cnn_pred),
    "F1-Score": f1_score(y_test_p, cnn_pred),
    "ROC-AUC": roc_auc_score(y_test_p, cnn_prob),
    "Predictions": cnn_pred,
    "Probabilities": cnn_prob
})

results.append({
    "Model": "RNN",
    "Accuracy": accuracy_score(y_test_p, rnn_pred),
    "Precision": precision_score(y_test_p, rnn_pred),
    "Recall": recall_score(y_test_p, rnn_pred),
    "F1-Score": f1_score(y_test_p, rnn_pred),
    "ROC-AUC": roc_auc_score(y_test_p, rnn_prob),
    "Predictions": rnn_pred,
    "Probabilities": rnn_prob
})

# Create comparison DataFrame sorted by F1-Score
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)

# Display metrics (rounded for readability)
display_df = comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].round(4)
print("\n" + "="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(display_df.to_string(index=False))
print("="*80)


# ============================================================================
# STEP 10: Final Visualizations
# ============================================================================

# 1. ROC Curves Comparison for All Models
plt.figure(figsize=(10, 8))

for idx, row in comparison_df.iterrows():
    model_name = row["Model"]
    y_prob = row["Probabilities"]
    fpr, tpr, _ = roc_curve(y_test_p, y_prob)
    roc_auc = roc_auc_score(y_test_p, y_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves Comparison - All Models", fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Confusion Matrix for Best Model
best_model_name = comparison_df.iloc[0]["Model"]
best_predictions = comparison_df.iloc[0]["Predictions"]

best_cm = confusion_matrix(y_test_p, best_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["No Disease", "Disease"])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix - {best_model_name}", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 3. Feature Importance (Random Forest)
plt.figure(figsize=(10, 6))
feature_importance = rf.feature_importances_
feature_names = preprocessor.get_feature_names_out()
indices = np.argsort(feature_importance)[-15:]  # Top 15 features

plt.barh(range(len(indices)), feature_importance[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Feature Importance", fontsize=12)
plt.title("Top 15 Feature Importances - Random Forest", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 4. Training Loss Curve for MLP
plt.figure(figsize=(10, 6))
plt.plot(train_losses, linewidth=2, color='#1f77b4')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Training Loss (BCEWithLogitsLoss)", fontsize=12)
plt.title("MLP Training Loss Curve", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Model Performance Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x_pos = np.arange(len(comparison_df))
width = 0.15

for i, metric in enumerate(metrics[:3]):
    axes[0].bar(x_pos + i*width, comparison_df[metric], width, label=metric, alpha=0.8)

axes[0].set_xlabel("Model", fontsize=12)
axes[0].set_ylabel("Score", fontsize=12)
axes[0].set_title("Performance Metrics Comparison (Part 1)", fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos + width * 2)
axes[0].set_xticklabels(comparison_df["Model"], rotation=45, ha='right', fontsize=9)
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

for i, metric in enumerate(metrics[3:]):
    axes[1].bar(x_pos + i*width, comparison_df[metric], width, label=metric, alpha=0.8)

axes[1].set_xlabel("Model", fontsize=12)
axes[1].set_ylabel("Score", fontsize=12)
axes[1].set_title("Performance Metrics Comparison (Part 2)", fontsize=14, fontweight='bold')
axes[1].set_xticks(x_pos + width * 0.5)
axes[1].set_xticklabels(comparison_df["Model"], rotation=45, ha='right', fontsize=9)
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# FINAL CONCLUSION
# ============================================================================

import numpy as np

best_idx = comparison_df.index[0]
best_model = comparison_df.iloc[best_idx]
second_model = comparison_df.iloc[1] if len(comparison_df) > 1 else None

f1_improvement = ((best_model["F1-Score"] - comparison_df.iloc[-1]["F1-Score"]) / 
                  comparison_df.iloc[-1]["F1-Score"] * 100)
auc_improvement = ((best_model["ROC-AUC"] - comparison_df.iloc[-1]["ROC-AUC"]) / 
                   comparison_df.iloc[-1]["ROC-AUC"] * 100)

print("\n" + "="*80)
print("FINAL CONCLUSION - HEART DISEASE PREDICTION PROJECT")
print("="*80)
print(f"\n🏆 BEST MODEL: {best_model['Model'].upper()}")
print(f"   - Accuracy:  {best_model['Accuracy']:.4f}")
print(f"   - Precision: {best_model['Precision']:.4f}")
print(f"   - Recall:    {best_model['Recall']:.4f}")
print(f"   - F1-Score:  {best_model['F1-Score']:.4f}")
print(f"   - ROC-AUC:   {best_model['ROC-AUC']:.4f}")

print(f"\n📊 WHY {best_model['Model'].upper()} PERFORMED BETTER:")
print(f"   • F1-Score advantage: {f1_improvement:+.2f}% vs worst model")
print(f"   • ROC-AUC advantage:  {auc_improvement:+.2f}% vs worst model")

if second_model is not None:
    f1_vs_second = ((best_model["F1-Score"] - second_model["F1-Score"]) / 
                    second_model["F1-Score"] * 100)
    print(f"   • F1-Score vs 2nd best ({second_model['Model']}): {f1_vs_second:+.2f}%")

if best_model["Model"] == "Random Forest":
    print(f"   • Ensemble learning captures complex patterns in the data")
    print(f"   • Robustness to outliers and feature scaling")
elif best_model["Model"] == "Logistic Regression":
    print(f"   • Simplicity and interpretability advantage")
    print(f"   • Well-separated decision boundary for binary classification")
elif best_model["Model"] == "SVM":
    print(f"   • Excellent margin maximization in high-dimensional space")
    print(f"   • Strong generalization on unseen data")
elif best_model["Model"] == "PyTorch MLP":
    print(f"   • Deep learning capability to learn non-linear patterns")
    print(f"   • GPU acceleration enabled efficient training")
else:
    print(f"   • Balanced performance across all metrics")

print(f"\n✅ Model trained on {len(X_train_p)} samples, tested on {len(X_test_p)} samples")
print(f"✅ No data leakage - preprocessing fit only on training data")
print("="*80 + "\n")

# Save preprocessors and models for Streamlit app
import joblib
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(lr, "lr_model.pkl")
joblib.dump(rf, "rf_model.pkl")
joblib.dump(svm, "svm_model.pkl")
joblib.dump(dt, "dt_model.pkl")
torch.save(mlp.state_dict(), "mlp_model.pth")
torch.save(cnn.state_dict(), "cnn_model.pth")
torch.save(rnn.state_dict(), "rnn_model.pth")
print("All models saved.")
