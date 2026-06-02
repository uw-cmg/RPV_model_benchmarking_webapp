import base64
import io
import json
import os
import uuid
from functools import lru_cache

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import models as rpv_models
from models import EnsembleNN_Jacobs26


PREDICTION_COLUMN = "Jacobs26 NN ensemble predicted TTS (degC)"
ERROR_BAR_COLUMN = "Jacobs26 NN ensemble error bars (degC)"
DOMAIN_COLUMN = "Jacobs26 NN ensemble domain d"
DOMAIN_THRESHOLD = 0.9

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

DISPLAY_COLUMNS = [
    PREDICTION_COLUMN,
    ERROR_BAR_COLUMN,
    DOMAIN_COLUMN,
    "Jacobs26 NN ensemble out of domain",
] + RAW_REQUIRED_COLUMNS

RESULT_CACHE = {}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


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
    model_folder = os.path.join(rpv_models.path, "model_files/Jacobs26/fullfit")
    model = predictor._rebuild_model(len(feature_columns), model_folder)
    with open(os.path.join(model_folder, "scaler_stats.json"), "r", encoding="utf-8") as f:
        scaler_stats = json.load(f)
    preprocessor = StandardScalerAdapter(
        feature_columns,
        mean=scaler_stats["mean"],
        scale=scaler_stats["scale"],
    )
    return predictor, feature_columns, model, preprocessor


def read_uploaded_table(upload):
    filename = (upload.filename or "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(upload)
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)
    raise ValueError("Upload a CSV or XLSX file.")


def prepare_input(df):
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    prepared = df.copy()
    for column in RAW_REQUIRED_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    invalid_numeric = [
        column for column in RAW_REQUIRED_COLUMNS if prepared[column].isna().any()
    ]
    if invalid_numeric:
        raise ValueError(
            "These required columns contain blank or non-numeric values: "
            + ", ".join(invalid_numeric)
        )

    nonpositive = [
        column
        for column in ["fluence_n_cm2", "flux_n_cm2_sec"]
        if (prepared[column] <= 0).any()
    ]
    if nonpositive:
        raise ValueError(
            "These columns must be positive because their log10 values are used: "
            + ", ".join(nonpositive)
        )

    prepared["log(fluence_n_cm2)"] = np.log10(prepared["fluence_n_cm2"].astype(float))
    prepared["log(flux_n_cm2_sec)"] = np.log10(
        prepared["flux_n_cm2_sec"].astype(float)
    )
    return prepared


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
    results["Jacobs26 NN ensemble out of domain"] = results[DOMAIN_COLUMN] > DOMAIN_THRESHOLD
    return results


def histogram_data_uri(predictions):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    bins = min(30, max(5, int(np.sqrt(len(predictions)))))
    ax.hist(predictions, bins=bins, color="#2f6f73", edgecolor="#f8fbfa", linewidth=0.8)
    ax.set_title("Predicted TTS Distribution")
    ax.set_xlabel("Predicted TTS (degC)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def cache_results(results):
    result_id = uuid.uuid4().hex
    payload = results.to_csv(index=False).encode("utf-8")
    RESULT_CACHE[result_id] = payload
    return result_id


@app.get("/health")
def health():
    return {"status": "ok", "model": "Jacobs26 NN"}


@app.get("/")
def index():
    return render_template(
        "index.html",
        required_columns=RAW_REQUIRED_COLUMNS,
        result=None,
        error=None,
    )


@app.post("/predict")
def predict():
    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return render_template(
            "index.html",
            required_columns=RAW_REQUIRED_COLUMNS,
            result=None,
            error="Choose a CSV or XLSX file.",
        ), 400

    try:
        input_df = prepare_input(read_uploaded_table(upload))
        results = predict_jacobs26(input_df)
    except Exception as exc:
        return render_template(
            "index.html",
            required_columns=RAW_REQUIRED_COLUMNS,
            result=None,
            error=str(exc),
        ), 400

    result_id = cache_results(results)
    table_columns = [column for column in DISPLAY_COLUMNS if column in results.columns]
    result = {
        "rows": len(results),
        "download_id": result_id,
        "plot_uri": histogram_data_uri(results[PREDICTION_COLUMN].to_numpy()),
        "records": results[table_columns].head(100).to_dict(orient="records"),
        "columns": table_columns,
        "truncated": len(results) > 100,
    }
    return render_template(
        "index.html",
        required_columns=RAW_REQUIRED_COLUMNS,
        result=result,
        error=None,
    )


@app.post("/api/predict")
def api_predict():
    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return jsonify({"error": "Choose a CSV or XLSX file."}), 400

    try:
        input_df = prepare_input(read_uploaded_table(upload))
        results = predict_jacobs26(input_df)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "model": "Jacobs26 NN",
            "domain_threshold": DOMAIN_THRESHOLD,
            "rows": len(results),
            "predictions": results.to_dict(orient="records"),
        }
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
        download_name="jacobs26_predictions.csv",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
