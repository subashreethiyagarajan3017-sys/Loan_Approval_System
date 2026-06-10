import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "loan_approval_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")

feature_columns = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "self_employed",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

numeric_features = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

categorical_features = ["self_employed"]

model = None
model_accuracy = None
load_error = None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def train_model() -> Pipeline:
    global model, model_accuracy

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset file not found.")

    df = pd.read_csv(DATA_PATH)
    df = clean_dataframe(df)

    if "self_employed" not in df.columns or "loan_status" not in df.columns:
        raise ValueError("Dataset is missing required columns.")

    df = df.drop(columns=["loan_id", "education", "residential_assets_value"], errors="ignore")
    df["self_employed"] = df["self_employed"].fillna("No").astype(str)

    X = df[feature_columns].copy()
    y = df["loan_status"].astype(str).copy()

    for column in numeric_features:
        X[column] = pd.to_numeric(X[column], errors="coerce").fillna(0)

    pipeline = build_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    model_accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    model = pipeline

    with open(MODEL_PATH, "wb") as handle:
        pickle.dump({"pipeline": pipeline, "accuracy": model_accuracy}, handle)

    return pipeline


def load_model() -> Pipeline:
    global model, model_accuracy

    if model is not None:
        return model

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as handle:
            data = pickle.load(handle)
            model = data["pipeline"]
            model_accuracy = data.get("accuracy")
            return model

    return train_model()


def prepare_model() -> bool:
    global load_error
    try:
        load_model()
        return True
    except Exception as exc:
        load_error = str(exc)
        return False


@app.route("/", methods=["GET"])
def home():
    dataset_found = os.path.exists(DATA_PATH)
    prepare_model()

    return render_template(
        "index.html",
        dataset_found=dataset_found,
        model_accuracy=model_accuracy,
        load_error=load_error,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify(message="Use POST with JSON or form data to get predictions. Submit to /predict with the required fields."), 200

    if not prepare_model() or model is None:
        return jsonify(error="Model is not available for prediction."), 500

    data = request.get_json(silent=True)
    if not data:
        data = request.form.to_dict(flat=True)

    if not data:
        return jsonify(error="Invalid input. Send JSON body or form data."), 400

    missing_values = [col for col in feature_columns if col not in data or str(data[col]).strip() == ""]
    if missing_values:
        return jsonify(error=f"Missing values for: {', '.join(missing_values)}"), 400

    try:
        if isinstance(data, dict):
            data = {key: value for key, value in data.items()}

        if "self_employed" in data:
            raw_emp = str(data["self_employed"]).strip().lower()
            if raw_emp in ["1", "yes", "true", "y"]:
                data["self_employed"] = "Yes"
            elif raw_emp in ["0", "no", "false", "n"]:
                data["self_employed"] = "No"
            else:
                data["self_employed"] = str(data["self_employed"])

        sample = pd.DataFrame([data], columns=feature_columns)
        for column in numeric_features:
            sample[column] = pd.to_numeric(sample[column], errors="coerce").fillna(0)
        sample["self_employed"] = sample["self_employed"].astype(str)

        result = model.predict(sample)[0]
        prediction = "Approved" if str(result).strip().lower() in ["approved", "1", "yes", "true"] else "Rejected"

        return jsonify(prediction=prediction, model_accuracy=model_accuracy)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
