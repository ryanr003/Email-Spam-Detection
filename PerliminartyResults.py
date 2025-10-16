import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
columns = [f"feature_{i}" for i in range(57)] + ["label"]
data = pd.read_csv(url, header=None, names=columns)

# Split into features and labels
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Labels (0: not spam, 1: spam)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naïve Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)

# Train Neural Network
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)

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

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print results
print(results_df)
