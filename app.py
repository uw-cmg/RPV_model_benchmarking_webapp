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
DEFAULT_SELECTED_MODELS = ["Jacobs26"]

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


class PortableGradientBoostingRegressor:
    def __init__(self, init_constant, learning_rate, trees):
        self.init_constant = float(init_constant)
        self.learning_rate = float(learning_rate)
        self.trees = trees

    @classmethod
    def load(cls, filename):
        payload = np.load(filename, allow_pickle=False)
        n_estimators = int(payload["n_estimators"][0])
        trees = []
        for idx in range(n_estimators):
            trees.append(
                {
                    "children_left": payload[f"children_left_{idx}"],
                    "children_right": payload[f"children_right_{idx}"],
                    "feature": payload[f"feature_{idx}"],
                    "threshold": payload[f"threshold_{idx}"],
                    "value": payload[f"value_{idx}"],
                }
            )
        return cls(
            init_constant=payload["init_constant"][0],
            learning_rate=payload["learning_rate"][0],
            trees=trees,
        )

    def predict(self, values):
        values = np.asarray(values, dtype=float)
        predictions = np.full(values.shape[0], self.init_constant, dtype=float)
        for tree in self.trees:
            predictions += self.learning_rate * self._predict_tree(tree, values)
        return predictions

    @staticmethod
    def _predict_tree(tree, values):
        tree_predictions = np.empty(values.shape[0], dtype=float)
        children_left = tree["children_left"]
        children_right = tree["children_right"]
        features = tree["feature"]
        thresholds = tree["threshold"]
        node_values = tree["value"]

        for row_index, row in enumerate(values):
            node = 0
            while children_left[node] != -1:
                feature = features[node]
                if row[feature] <= thresholds[node]:
                    node = children_left[node]
                else:
                    node = children_right[node]
            tree_predictions[row_index] = node_values[node]
        return tree_predictions


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
    model = PortableGradientBoostingRegressor.load(
        os.path.join(model_folder, "GradientBoostingRegressor_portable.npz")
    )
    return feature_columns, model, preprocessor


def read_uploaded_table(upload):
    filename = (upload.filename or "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(upload)
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)
    raise ValueError("Upload a CSV or XLSX file.")


def selected_models():
    model_names = request.form.getlist("models")
    if not model_names and request.form.get("models_submitted"):
        raise ValueError("Choose at least one model.")
    if not model_names:
        legacy_model = request.form.get("model")
        model_names = [legacy_model] if legacy_model else DEFAULT_SELECTED_MODELS

    unsupported = [model_name for model_name in model_names if model_name not in MODEL_SPECS]
    if unsupported:
        raise ValueError("Unsupported model: " + ", ".join(unsupported))

    return [model_name for model_name, _label in MODEL_OPTIONS if model_name in model_names]


def required_columns_for_models(model_names):
    required_columns = []
    for model_name in model_names:
        for column in MODEL_SPECS[model_name]["required"]:
            if column not in required_columns:
                required_columns.append(column)
    return required_columns


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


def prepare_input_for_models(df, model_names):
    required_columns = required_columns_for_models(model_names)
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
        if column in required_columns and (prepared[column] <= 0).any():
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


def predict_models(df, model_names):
    results = df.copy()
    for model_name in model_names:
        model_results = predict_model(results, model_name)
        for column in MODEL_SPECS[model_name]["outputs"]:
            if column in model_results.columns:
                results[column] = model_results[column]
    return results


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


def output_columns(results, uploaded_columns, model_names):
    output_columns_for_models = []
    for model_name in model_names:
        for column in MODEL_SPECS[model_name]["outputs"]:
            if column in results.columns and column not in output_columns_for_models:
                output_columns_for_models.append(column)

    original_columns = [column for column in uploaded_columns if column in results.columns]
    return original_columns + output_columns_for_models


def cache_results(results):
    result_id = uuid.uuid4().hex
    payload = results.to_csv(index=False).encode("utf-8")
    RESULT_CACHE[result_id] = payload
    return result_id


def render_prediction(uploaded_df, model_names):
    uploaded_columns = list(uploaded_df.columns)
    input_df = prepare_input_for_models(uploaded_df, model_names)
    results = predict_models(input_df, model_names)
    table_columns = output_columns(results, uploaded_columns, model_names)
    ordered_results = results[table_columns]
    result_id = cache_results(ordered_results)
    return {
        "rows": len(results),
        "download_id": result_id,
        "records": ordered_results.head(100).to_dict(orient="records"),
        "columns": table_columns,
        "truncated": len(results) > 100,
        "selected_models": model_names,
    }


@app.get("/health")
def health():
    return {"status": "ok", "models": [value for value, _label in MODEL_OPTIONS]}


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_options=MODEL_OPTIONS,
        model_specs=MODEL_SPECS,
        selected_models=DEFAULT_SELECTED_MODELS,
        required_columns=required_columns_for_models(DEFAULT_SELECTED_MODELS),
        result=None,
        error=None,
    )


@app.post("/predict")
def predict():
    try:
        model_names = selected_models()
        upload = request.files.get("file")
        if upload is None or upload.filename == "":
            raise ValueError("Choose a CSV or XLSX file.")
        uploaded_df = read_uploaded_table(upload)
        result = render_prediction(uploaded_df, model_names)
    except Exception as exc:
        model_names = request.form.getlist("models") or DEFAULT_SELECTED_MODELS
        model_names = [model_name for model_name in model_names if model_name in MODEL_SPECS]
        if not model_names:
            model_names = DEFAULT_SELECTED_MODELS
        return render_template(
            "index.html",
            model_options=MODEL_OPTIONS,
            model_specs=MODEL_SPECS,
            selected_models=model_names,
            required_columns=required_columns_for_models(model_names),
            result=None,
            error=str(exc),
        ), 400

    return render_template(
        "index.html",
        model_options=MODEL_OPTIONS,
        model_specs=MODEL_SPECS,
        selected_models=model_names,
        required_columns=required_columns_for_models(model_names),
        result=result,
        error=None,
    )


@app.post("/predict-example")
def predict_example():
    try:
        model_names = selected_models()
        uploaded_df = pd.read_excel(EXAMPLE_FILE)
        result = render_prediction(uploaded_df, model_names)
    except Exception as exc:
        model_names = request.form.getlist("models") or DEFAULT_SELECTED_MODELS
        model_names = [model_name for model_name in model_names if model_name in MODEL_SPECS]
        if not model_names:
            model_names = DEFAULT_SELECTED_MODELS
        return render_template(
            "index.html",
            model_options=MODEL_OPTIONS,
            model_specs=MODEL_SPECS,
            selected_models=model_names,
            required_columns=required_columns_for_models(model_names),
            result=None,
            error=str(exc),
        ), 400

    return render_template(
        "index.html",
        model_options=MODEL_OPTIONS,
        model_specs=MODEL_SPECS,
        selected_models=model_names,
        required_columns=required_columns_for_models(model_names),
        result=result,
        error=None,
    )


@app.post("/api/predict")
def api_predict():
    try:
        model_names = selected_models()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return jsonify({"error": "Choose a CSV or XLSX file."}), 400

    try:
        uploaded_df = read_uploaded_table(upload)
        uploaded_columns = list(uploaded_df.columns)
        input_df = prepare_input_for_models(uploaded_df, model_names)
        results = predict_models(input_df, model_names)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    columns = output_columns(results, uploaded_columns, model_names)
    ordered_results = results[columns]
    return jsonify(
        {
            "models": [dict(MODEL_OPTIONS)[model_name] for model_name in model_names],
            "selected_models": model_names,
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
