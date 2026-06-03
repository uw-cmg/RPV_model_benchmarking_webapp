import io
import json
import os
import uuid
from functools import lru_cache

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from models import E900, EONY, EnsembleNN_Jacobs23, EnsembleNN_Jacobs26
import joblib


PREDICTION_COLUMN = "Jacobs26 NN ensemble predicted TTS (degC)"
ERROR_BAR_COLUMN = "Jacobs26 NN ensemble error bars (degC)"
DOMAIN_COLUMN = "Jacobs26 NN ensemble domain d"
DOMAIN_THRESHOLD = 0.9
OUT_OF_DOMAIN_COLUMN = "Jacobs26 NN ensemble out of domain"

RAW_REQUIRED_COLUMNS = [
    "temperature_C",
    "wt_percent_Cu",
    "wt_percent_Ni",
    "wt_percent_Mn",
    "wt_percent_P",
    "wt_percent_Si",
    "wt_percent_C",
    "fluence_n_cm2",
    "flux_n_cm2_sec",
]

MODEL_INPUT_COLUMNS = RAW_REQUIRED_COLUMNS + [
    "log(fluence_n_cm2)",
    "log(flux_n_cm2_sec)",
]

MODEL_OPTIONS = [
    ("Jacobs26", "Jacobs26 NN"),
    ("Jacobs23", "Jacobs23 NN"),
    ("GBR", "GBR"),
    ("EONY", "EONY"),
    ("E900", "E900"),
]

MODEL_SPECS = {
    "Jacobs26": {
        "required": RAW_REQUIRED_COLUMNS,
        "inputs": MODEL_INPUT_COLUMNS,
        "outputs": [PREDICTION_COLUMN, ERROR_BAR_COLUMN, DOMAIN_COLUMN, OUT_OF_DOMAIN_COLUMN],
    },
    "Jacobs23": {
        "required": RAW_REQUIRED_COLUMNS,
        "inputs": RAW_REQUIRED_COLUMNS[:7] + ["log(fluence_n_cm2)", "log(flux_n_cm2_sec)"],
        "outputs": [
            "Jacobs23 NN ensemble predicted TTS (degC)",
            "Jacobs23 NN ensemble error bars (degC)",
        ],
    },
    "GBR": {
        "required": [
            "temperature_C",
            "wt_percent_Cu",
            "wt_percent_Ni",
            "wt_percent_Mn",
            "wt_percent_P",
            "fluence_n_cm2",
            "flux_n_cm2_sec",
            "Product Form",
            "Reactor Type",
        ],
        "inputs": [
            "temperature_C",
            "wt_percent_Cu",
            "wt_percent_Ni",
            "wt_percent_Mn",
            "wt_percent_P",
            "fluence_n_cm2",
            "flux_n_cm2_sec",
            "Product Form",
            "Reactor Type",
        ],
        "outputs": ["GBR predicted TTS (degC)"],
    },
    "EONY": {
        "required": [
            "Product Form",
            "temperature_C",
            "wt_percent_Cu",
            "wt_percent_Ni",
            "wt_percent_Mn",
            "wt_percent_P",
            "flux_n_cm2_sec",
            "fluence_n_cm2",
        ],
        "inputs": [
            "Product Form",
            "temperature_C",
            "wt_percent_Cu",
            "wt_percent_Ni",
            "wt_percent_Mn",
            "wt_percent_P",
            "flux_n_cm2_sec",
            "fluence_n_cm2",
        ],
        "outputs": ["EONY predicted TTS (degC)"],
    },
    "E900": {
        "required": [
            "Product Form",
            "temperature_C",
            "wt_percent_Cu",
            "wt_percent_Ni",
            "wt_percent_Mn",
            "wt_percent_P",
            "flux_n_cm2_sec",
            "fluence_n_cm2",
        ],
        "inputs": [
            "Product Form",
            "temperature_C",
            "wt_percent_Cu",
            "wt_percent_Ni",
            "wt_percent_Mn",
            "wt_percent_P",
            "flux_n_cm2_sec",
            "fluence_n_cm2",
        ],
        "outputs": ["E900 predicted TTS (degC)"],
    },
}

NUMERIC_COLUMNS = {
    "temperature_C",
    "wt_percent_Cu",
    "wt_percent_Ni",
    "wt_percent_Mn",
    "wt_percent_P",
    "wt_percent_Si",
    "wt_percent_C",
    "fluence_n_cm2",
    "flux_n_cm2_sec",
}

RESULT_CACHE = {}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_FILE = os.path.join(APP_ROOT, "RPV_test.xlsx")


class StandardScalerAdapter:
    def __init__(self, feature_columns, mean, scale):
        self.feature_columns = feature_columns
        self.mean = np.array(mean, dtype=float)
        self.scale = np.array(scale, dtype=float)

    def transform(self, df):
        values = df[self.feature_columns].to_numpy(dtype=float)
        scaled = (values - self.mean) / self.scale
        return pd.DataFrame(scaled, columns=self.feature_columns, index=df.index)


@lru_cache(maxsize=1)
def load_jacobs26_assets():
    predictor = EnsembleNN_Jacobs26()
    feature_columns = predictor._features()
    model_folder = os.path.join(APP_ROOT, "model_files/Jacobs26/fullfit")
    model = predictor._rebuild_model(len(feature_columns), model_folder)
    with open(os.path.join(model_folder, "scaler_stats.json"), "r", encoding="utf-8") as f:
        scaler_stats = json.load(f)
    preprocessor = StandardScalerAdapter(
        feature_columns,
        mean=scaler_stats["mean"],
        scale=scaler_stats["scale"],
    )
    return predictor, feature_columns, model, preprocessor


@lru_cache(maxsize=1)
def load_jacobs23_assets():
    predictor = EnsembleNN_Jacobs23()
    feature_columns = predictor._features()
    model_folder = os.path.join(APP_ROOT, "model_files/Jacobs23/fullfit")
    model = predictor._rebuild_model(len(feature_columns), model_folder)
    with open(os.path.join(model_folder, "scaler_stats.json"), "r", encoding="utf-8") as f:
        scaler_stats = json.load(f)
    preprocessor = StandardScalerAdapter(
        feature_columns,
        mean=scaler_stats["mean"],
        scale=scaler_stats["scale"],
    )
    return predictor, feature_columns, model, preprocessor


@lru_cache(maxsize=1)
def load_gbr_assets():
    model_folder = os.path.join(APP_ROOT, "model_files/GBR/fullfit")
    feature_columns = [
        "temperature_C",
        "wt_percent_Cu",
        "wt_percent_Ni",
        "wt_percent_Mn",
        "wt_percent_P",
        "fluence_n_cm2",
        "flux_n_cm2_sec",
        "Product Form_0",
        "Product Form_1",
        "Product Form_2",
        "Product Form_3",
        "Product Form_4",
        "Product Form_5",
        "Reactor Type_0",
        "Reactor Type_1",
    ]
    with open(os.path.join(model_folder, "scaler_stats.json"), "r", encoding="utf-8") as f:
        scaler_stats = json.load(f)
    preprocessor = StandardScalerAdapter(
        feature_columns,
        mean=scaler_stats["mean"],
        scale=scaler_stats["scale"],
    )
    model = joblib.load(os.path.join(model_folder, "GradientBoostingRegressor_sklearn.pkl"))
    return feature_columns, model, preprocessor


def read_uploaded_table(upload):
    filename = (upload.filename or "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(upload)
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)
    raise ValueError("Upload a CSV or XLSX file.")


def selected_model():
    model_name = request.form.get("model", "Jacobs26")
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported model: {model_name}")
    return model_name


def prepare_input(df, model_name):
    required_columns = MODEL_SPECS[model_name]["required"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    prepared = df.copy()
    numeric_required = [column for column in required_columns if column in NUMERIC_COLUMNS]
    for column in numeric_required:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    invalid_numeric = [
        column for column in numeric_required if prepared[column].isna().any()
    ]
    if invalid_numeric:
        raise ValueError(
            "These required columns contain blank or non-numeric values: "
            + ", ".join(invalid_numeric)
        )

    nonpositive = []
    for column in ["fluence_n_cm2", "flux_n_cm2_sec"]:
        if column in prepared.columns and column in required_columns and (prepared[column] <= 0).any():
            nonpositive.append(column)
    if nonpositive:
        raise ValueError(
            "These columns must be positive because their log10 values are used: "
            + ", ".join(nonpositive)
        )

    if "fluence_n_cm2" in prepared.columns:
        prepared["log(fluence_n_cm2)"] = np.log10(prepared["fluence_n_cm2"].astype(float))
    if "flux_n_cm2_sec" in prepared.columns:
        prepared["log(flux_n_cm2_sec)"] = np.log10(
            prepared["flux_n_cm2_sec"].astype(float)
        )
    return prepared


def predict_model(df, model_name):
    if model_name == "Jacobs26":
        return predict_jacobs26(df)
    if model_name == "Jacobs23":
        return predict_jacobs23(df)
    if model_name == "GBR":
        return predict_gbr(df)
    if model_name == "EONY":
        return EONY().predict(df.copy())[1]
    if model_name == "E900":
        return E900().predict(df.copy())[1]
    raise ValueError(f"Unsupported model: {model_name}")


def predict_jacobs26(df):
    predictor, feature_columns, model, preprocessor = load_jacobs26_assets()
    preds, ebars, domains = predictor._get_preds_ebars(
        model,
        df[feature_columns],
        preprocessor,
        return_ebars=True,
        return_domains=True,
    )

    results = df.copy()
    results[PREDICTION_COLUMN] = preds
    results[ERROR_BAR_COLUMN] = ebars
    results[DOMAIN_COLUMN] = domains
    results[OUT_OF_DOMAIN_COLUMN] = results[DOMAIN_COLUMN] > DOMAIN_THRESHOLD
    return results


def predict_jacobs23(df):
    predictor, feature_columns, model, preprocessor = load_jacobs23_assets()
    preds, ebars = predictor._get_preds_ebars(
        model,
        df[feature_columns],
        preprocessor,
        return_ebars=True,
    )

    results = df.copy()
    results["Jacobs23 NN ensemble predicted TTS (degC)"] = preds
    results["Jacobs23 NN ensemble error bars (degC)"] = ebars
    return results


def predict_gbr(df):
    feature_columns, model, preprocessor = load_gbr_assets()
    results = df.copy()
    encoded = pd.DataFrame(index=results.index)
    encoded["temperature_C"] = results["temperature_C"]
    encoded["wt_percent_Cu"] = results["wt_percent_Cu"]
    encoded["wt_percent_Ni"] = results["wt_percent_Ni"]
    encoded["wt_percent_Mn"] = results["wt_percent_Mn"]
    encoded["wt_percent_P"] = results["wt_percent_P"]
    encoded["fluence_n_cm2"] = results["fluence_n_cm2"]
    encoded["flux_n_cm2_sec"] = results["flux_n_cm2_sec"]

    product_forms = results["Product Form"].astype(str)
    encoded["Product Form_0"] = (product_forms == "F").astype(int)
    encoded["Product Form_1"] = (product_forms == "HAZ").astype(int)
    encoded["Product Form_2"] = (product_forms == "P").astype(int)
    encoded["Product Form_3"] = (product_forms == "SRM").astype(int)
    encoded["Product Form_4"] = np.where(product_forms == "W", 4, 0)
    encoded["Product Form_5"] = (product_forms == "PCE").astype(int)

    reactor_types = results["Reactor Type"].astype(str)
    encoded["Reactor Type_0"] = (reactor_types == "BWR").astype(int)
    encoded["Reactor Type_1"] = (reactor_types == "PWR").astype(int)

    preds = model.predict(preprocessor.transform(encoded[feature_columns]))
    results["GBR predicted TTS (degC)"] = preds
    return results


def output_columns(results, uploaded_columns, model_name):
    spec = MODEL_SPECS[model_name]
    current_columns = [
        column for column in spec["outputs"] + spec["required"] if column in results.columns
    ]
    metadata_columns = [
        column
        for column in uploaded_columns
        if column not in current_columns and column not in spec["inputs"]
    ]
    return current_columns + metadata_columns


def cache_results(results):
    result_id = uuid.uuid4().hex
    payload = results.to_csv(index=False).encode("utf-8")
    RESULT_CACHE[result_id] = payload
    return result_id


@app.get("/health")
def health():
    return {"status": "ok", "models": [value for value, _label in MODEL_OPTIONS]}


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_options=MODEL_OPTIONS,
        model_specs=MODEL_SPECS,
        selected_model="Jacobs26",
        required_columns=MODEL_SPECS["Jacobs26"]["required"],
        result=None,
        error=None,
    )


@app.post("/predict")
def predict():
    model_name = selected_model()
    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return render_template(
            "index.html",
            model_options=MODEL_OPTIONS,
            model_specs=MODEL_SPECS,
            selected_model=model_name,
            required_columns=MODEL_SPECS[model_name]["required"],
            result=None,
            error="Choose a CSV or XLSX file.",
        ), 400

    try:
        uploaded_df = read_uploaded_table(upload)
        uploaded_columns = list(uploaded_df.columns)
        input_df = prepare_input(uploaded_df, model_name)
        results = predict_model(input_df, model_name)
    except Exception as exc:
        return render_template(
            "index.html",
            model_options=MODEL_OPTIONS,
            model_specs=MODEL_SPECS,
            selected_model=model_name,
            required_columns=MODEL_SPECS[model_name]["required"],
            result=None,
            error=str(exc),
        ), 400

    table_columns = output_columns(results, uploaded_columns, model_name)
    ordered_results = results[table_columns]
    result_id = cache_results(ordered_results)
    result = {
        "rows": len(results),
        "download_id": result_id,
        "records": ordered_results.head(100).to_dict(orient="records"),
        "columns": table_columns,
        "truncated": len(results) > 100,
    }
    return render_template(
        "index.html",
        model_options=MODEL_OPTIONS,
        model_specs=MODEL_SPECS,
        selected_model=model_name,
        required_columns=MODEL_SPECS[model_name]["required"],
        result=result,
        error=None,
    )


@app.post("/predict-example")
def predict_example():
    model_name = selected_model()
    try:
        uploaded_df = pd.read_excel(EXAMPLE_FILE)
        uploaded_columns = list(uploaded_df.columns)
        input_df = prepare_input(uploaded_df, model_name)
        results = predict_model(input_df, model_name)
    except Exception as exc:
        return render_template(
            "index.html",
            model_options=MODEL_OPTIONS,
            model_specs=MODEL_SPECS,
            selected_model=model_name,
            required_columns=MODEL_SPECS[model_name]["required"],
            result=None,
            error=str(exc),
        ), 400

    table_columns = output_columns(results, uploaded_columns, model_name)
    ordered_results = results[table_columns]
    result_id = cache_results(ordered_results)
    result = {
        "rows": len(results),
        "download_id": result_id,
        "records": ordered_results.head(100).to_dict(orient="records"),
        "columns": table_columns,
        "truncated": len(results) > 100,
    }
    return render_template(
        "index.html",
        model_options=MODEL_OPTIONS,
        model_specs=MODEL_SPECS,
        selected_model=model_name,
        required_columns=MODEL_SPECS[model_name]["required"],
        result=result,
        error=None,
    )


@app.post("/api/predict")
def api_predict():
    model_name = request.form.get("model", "Jacobs26")
    if model_name not in MODEL_SPECS:
        return jsonify({"error": f"Unsupported model: {model_name}"}), 400
    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return jsonify({"error": "Choose a CSV or XLSX file."}), 400

    try:
        uploaded_df = read_uploaded_table(upload)
        uploaded_columns = list(uploaded_df.columns)
        input_df = prepare_input(uploaded_df, model_name)
        results = predict_model(input_df, model_name)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    columns = output_columns(results, uploaded_columns, model_name)
    ordered_results = results[columns]
    return jsonify(
        {
            "model": dict(MODEL_OPTIONS)[model_name],
            "selected_model": model_name,
            "domain_threshold": DOMAIN_THRESHOLD,
            "rows": len(results),
            "predictions": ordered_results.to_dict(orient="records"),
        }
    )


@app.get("/example")
def download_example():
    return send_file(
        EXAMPLE_FILE,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="RPV_test.xlsx",
    )


@app.get("/download/<result_id>")
def download(result_id):
    payload = RESULT_CACHE.get(result_id)
    if payload is None:
        return "Result not found. Re-run the prediction.", 404

    return send_file(
        io.BytesIO(payload),
        mimetype="text/csv",
        as_attachment=True,
        download_name="rpv_model_predictions.csv",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
