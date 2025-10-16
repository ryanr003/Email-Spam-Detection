#Final Report Code: Group 44

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
columns = [f"feature_{i}" for i in range(57)] + ["label"]
data = pd.read_csv(url, header=None, names=columns)



# Split into features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)



# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Train Naïve Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_proba = nb.predict_proba(X_test)[:, 1]



# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_reg_pred = log_reg.predict(X_test_scaled)
log_reg_proba = log_reg.predict_proba(X_test_scaled)[:, 1]



# Train Neural Network with Grid Search
nn_params = {
    'hidden_layer_sizes': [(100,), (100, 50)],
    'alpha': [0.0001, 0.001],
}
nn_grid = GridSearchCV(MLPClassifier(max_iter=300, random_state=42), nn_params, cv=3, scoring='f1')
nn_grid.fit(X_train_scaled, y_train)
nn = nn_grid.best_estimator_
nn_pred = nn.predict(X_test_scaled)
nn_proba = nn.predict_proba(X_test_scaled)[:, 1]



# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }



# Collect results
results = [
    evaluate_model(y_test, nb_pred, "Naïve Bayes"),
    evaluate_model(y_test, log_reg_pred, "Logistic Regression"),
    evaluate_model(y_test, nn_pred, "Neural Network")
]
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:\n")
print(results_df)



# Plot confusion matrices
def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_conf_matrix(y_test, nb_pred, "Naïve Bayes")
plot_conf_matrix(y_test, log_reg_pred, "Logistic Regression")
plot_conf_matrix(y_test, nn_pred, "Neural Network")



# Plot ROC Curves
def plot_roc(y_true, probs, label):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

plt.figure(figsize=(8, 6))
plot_roc(y_test, nb_proba, "Naïve Bayes")
plot_roc(y_test, log_reg_proba, "Logistic Regression")
plot_roc(y_test, nn_proba, "Neural Network")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid()
plt.show()