from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from flask import Flask, request, jsonify
import pandas as pd
import os
import numpy as np


app = Flask(__name__)

data = None
model = None

@app.route('/')
def home():
    return "Welcome to the Manufacturing Predictive Analysis API!"

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/upload', methods=['POST'])
def upload_data():
    global data
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request!"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected!"}), 400

    try:
        data = pd.read_csv(file)
        return jsonify({"message": "Data uploaded successfully!", "data_preview": data.head().to_dict()}), 200
    except Exception as e:
        return jsonify({"error": f"Error reading file: {str(e)}"}), 500

@app.route('/train', methods=['POST'])
def train_model():
    global data, model
    if data is None:
        return jsonify({"error": "No data uploaded!"}), 400

    try:
        numeric_data = data.select_dtypes(include=[float, int])

        X = numeric_data.drop(columns=["Downtime_Flag"], errors="ignore")
        y = data["Downtime_Flag"]

    except KeyError as e:
        return jsonify({"error": f"Missing required column: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Data preprocessing error: {str(e)}"}), 500

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "AdaBoost": AdaBoostClassifier()
    }

    param_grids = {
        "DecisionTree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
        "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10, None]},
        "LogisticRegression": {"C": [0.1, 1.0, 10]},
        "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]}
    }

    best_model = None
    best_params = None
    best_score_diff = float('inf')
    best_model_name = None

    for model_name, clf in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])

        param_grid = {
            f"classifier__{key}": value for key, value in param_grids[model_name].items()
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        train_accuracy = accuracy_score(y_train, grid_search.best_estimator_.predict(X_train))
        test_accuracy = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
        score_diff = abs(train_accuracy - test_accuracy)

        if score_diff < best_score_diff:
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score_diff = score_diff
            best_model_name = model_name

    if not os.path.exists('model'):
        os.makedirs('model')
    model = best_model
    joblib.dump(model, 'model/model.pkl')

    return jsonify({
        "message": "Model trained successfully!",
        "best_model": best_model_name,
        "best_params": best_params,
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "train_test_accuracy_diff": round(best_score_diff, 4)
    }), 200



@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        if not os.path.exists('model/model.pkl'):
            return jsonify({"error": "No trained model found!"}), 400
        model = joblib.load('model/model.pkl')

    try:
        input_data = request.json
        features = [[input_data["Temperature"], input_data["Run_Time"]]]
    except (KeyError, TypeError):
        return jsonify({"error": "Invalid JSON input or missing required keys: 'Temperature' and 'Run_Time'"}), 400

    try:
        prediction = model.predict(features)[0]  # Get the first prediction
        prediction = str(prediction)  # Ensure it's JSON serializable

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(features).max()
            confidence = float(confidence)  # Ensure it's JSON serializable

        response = {"Downtime": prediction}
        if confidence is not None:
            response["Confidence"] = round(confidence, 2)

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(model, 'model/model.pkl')


if __name__ == '__main__':
    app.run(debug=True)