import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# --- Helper Function for Clean Reports ---
def evaluate_model(name, model, X_test_data, y_test_data):
    prediction = model.predict(X_test_data)
    print("=" * 50)
    print(f"Model: {name}")
    print("=" * 50)
    print("Accuracy :", accuracy_score(y_test_data, prediction))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_data, prediction))
    print("\nClassification Report:")
    print(classification_report(y_test_data, prediction))
    print("\n")
    return accuracy_score(y_test_data, prediction)

# --- 1. Load Data ---
# Change line 36-37 to this:
DATASET_PATH = r"dataset\heart_data_set.csv"
heartData = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("Dataset Information")
print("=" * 60)
heartData.info()

# --- 2. Data Splitting & Feature Scaling ---
X = heartData.drop("target", axis=1)
y = heartData["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Crucial: Scale the data properly for distance/gradient-based algorithms
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Model Training & Optimization ---

# A. K-Nearest Neighbors (KNN)
knn_scores = []
for k in range(2, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    score = accuracy_score(y_test, model.predict(X_test_scaled))
    knn_scores.append(score)

best_k = np.argmax(knn_scores) + 2
print(f"[KNN] Optimal K chosen: {best_k}")

# Re-train final KNN on scaled data
knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier.fit(X_train_scaled, y_train)

# B. Support Vector Machine (SVM)
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svc_scores = []
for kernel in kernels:
    # SVM highly benefits from scaled inputs
    svc = SVC(kernel=kernel, random_state=42)
    svc.fit(X_train_scaled, y_train)
    svc_scores.append(round(svc.score(X_test_scaled, y_test), 2))

# Select the best performing kernel (e.g., linear if using index 0)
svc_classifier = SVC(kernel='linear', random_state=42)
svc_classifier.fit(X_train_scaled, y_train)

# C. Decision Tree Classifier
dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt = DecisionTreeClassifier(max_features=i, random_state=42)
    dt.fit(X_train, y_train)  # Tree models don't strictly require scaling
    dt_scores.append(round(dt.score(X_test, y_test), 2))

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# D. Random Forest Classifier
estimators = [10, 20, 100, 200, 500]
rf_scores = []
for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    rf_scores.append(round(rf.score(X_test, y_test), 2))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# E. Logistic Regression
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train)

# --- 4. Comprehensive Evaluation ---
# Evaluate scaled models using X_test_scaled, tree models using X_test
evaluate_model("K-Nearest Neighbors", knn_classifier, X_test_scaled, y_test)
evaluate_model("Support Vector Machine", svc_classifier, X_test_scaled, y_test)
evaluate_model("Decision Tree", dt_classifier, X_test, y_test)
evaluate_model("Random Forest", rf_model, X_test, y_test)
evaluate_model("Logistic Regression", logistic_model, X_test_scaled, y_test)

# --- 5. Custom Live Test Sample (Fixed Scaling) ---
check_data1 = pd.DataFrame(
    [[52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3]],
    columns=X.columns
)
# Make sure to scale the test sample just like the training data!
check_data1_scaled = scaler.transform(check_data1)
sample_prediction = knn_classifier.predict(check_data1_scaled)
print(f"Sample Custom Test Prediction (KNN): {sample_prediction}\n")

# --- 6. Pickle Model Export (Cleaned up duplicates) ---
models_to_save = {
    "Random Forest": rf_model,
    "Logistic Regression": logistic_model,
    "Decision Tree": dt_classifier,
    "SVM": svc_classifier,
    "KNN": knn_classifier,
    "Scaler": scaler
}

with open("models.pkl", "wb") as f:
    pickle.dump(models_to_save, f)
print("All models and scaler successfully serialized to 'models.pkl'!")