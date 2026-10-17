import re
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, redirect

app = Flask(__name__)

# --- Load the saved models dictionary safely ---
try:
    with open('models.pkl', 'rb') as f:
        models_dict = pickle.load(f)
    print("Successfully loaded serialized models:", list(models_dict.keys()))
except FileNotFoundError:
    print("CRITICAL ERROR: 'models.pkl' not found. Please run your training script first.")
    models_dict = {}

@app.route('/', methods=['GET', 'POST'])
def hello():
    return render_template("index.html")

@app.route('/aboutUs', methods=['GET'])
def aboutUs():
    return render_template('aboutUs.html')

@app.route('/api', methods=['POST'])
def predict():
    # 1. Extract form values safely
    name = request.form.get('name', 'Guest')
    email = request.form.get('email', '')
    
    age = request.form.get('age', '0')
    fgender = request.form.get('gender', 'Male')
    cp = request.form.get('cp', '0')
    trestbps = request.form.get('trestbps', '120')
    chol = request.form.get('chol', '200')
    fbs = request.form.get('fbs', '0')
    restecg = request.form.get('restecg', '0')
    thalach = request.form.get('thalach', '150')
    exang = request.form.get('exang', '0')
    oldpeak = request.form.get('oldpeak', '0.0')
    slope = request.form.get('slope', '0')
    ca = request.form.get('ca', '0')
    thal = request.form.get('thal', '2')

    # 2. Handle empty inputs with smart defaults
    trestbps = 95 if trestbps == '' else int(trestbps)
    chol = 150 if chol == '' else int(chol)
    thalach = 72 if thalach == '' else int(thalach)
    oldpeak = 2.0 if oldpeak == '' else float(oldpeak) # Kept as float for decimal inputs

    # 3. Categorical encoding mapping to integers
    gender_encoded = 1 if fgender == "Male" else 0
    
    if thal == "Normal": thal_encoded = 0
    elif thal == "Fixed Defect": thal_encoded = 1
    else: thal_encoded = 2

    if restecg == "Normal": restecg_encoded = 0
    elif restecg == "STT Abnormality": restecg_encoded = 1
    else: restecg_encoded = 2

    exang_encoded = 1 if exang == "Yes" else 0

    # Explicit integer conversions for remaining features
    age = int(age)
    cp = int(cp)
    fbs = int(fbs)
    slope = int(slope)
    ca = int(ca)

    # 4. Build feature sets for different model categories
    # Order matches the training dataset exactly
    raw_features_list = [age, gender_encoded, cp, trestbps, chol, fbs, restecg_encoded, thalach, exang_encoded, oldpeak, slope, ca, thal_encoded]
    
    # Crucial: Scale features using the saved pipeline transformer
    feature_columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    features_df = pd.DataFrame([raw_features_list], columns=feature_columns)
    
    scaler = models_dict.get("Scaler")
    scaled_features = scaler.transform(features_df) if scaler else raw_features_list

    # 5. Iterative voting system across models
    results_display_dict = {}
    disease_votes = 0
    total_models = 0

    # Separate models based on whether they require scaled data
    models_to_evaluate = {
        "Random Forest": (models_dict.get("Random Forest"), False),
        "Decision Tree": (models_dict.get("Decision Tree"), False),
        "Logistic Regression": (models_dict.get("Logistic Regression"), True),
        "SVM": (models_dict.get("SVM"), True),
        "KNN": (models_dict.get("KNN"), True)
    }

    for model_name, (model_obj, requires_scaling) in models_to_evaluate.items():
        if model_obj is not None:
            # Pass correct scale block down to model execution
            input_features = scaled_features if requires_scaling else features_df
            res = model_obj.predict(input_features)
            
            if res[0] == 1:
                results_display_dict[model_name] = "High Chance of Heart Disease"
                disease_votes += 1
            else:
                results_display_dict[model_name] = "Low Chance of Heart Disease"
            total_models += 1

    # Calculate final ensemble baseline risk index (percentage)
    risk_percentage = round((disease_votes / total_models) * 100, 2) if total_models > 0 else 0.0

    # 6. Structuring final payload for render_template UI bindings
    input_data_summary = {
        "Age": age, "Gender": fgender, "Chest Pain Type": cp,
        "Resting Blood Pressure": trestbps, "Cholesterol Level": chol,
        "Fasting Blood Sugar > 120 mg/dl": fbs, "Resting ECG": restecg,
        "Max Heart Rate": thalach, "Exercise Induced Angina": exang,
        "ST Depression (Oldpeak)": oldpeak, "ST Slope": slope,
        "Major Vessels Count": ca, "Thalassemia Type": thal
    }

    personal_info = [name, email]
    responses = [input_data_summary, results_display_dict, personal_info, risk_percentage]
    
    return render_template("result.html", result=responses)

if __name__ == '__main__':
    app.run(port=5000, debug=True)