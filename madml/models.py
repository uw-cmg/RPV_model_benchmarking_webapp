import numpy as np
import pandas as pd


class combine:
    """Small prediction shim for MAD-ML combine objects saved in model.dill."""

    def predict(self, X):
        results = pd.DataFrame(index=getattr(X, "index", None))

        if hasattr(self, "ds_model"):
            results["d_pred"] = np.asarray(_predict_component(self.ds_model, X)).ravel()
        elif hasattr(self, "gs_model"):
            results["y_pred"] = np.asarray(self.gs_model.predict(X)).ravel()

        if hasattr(self, "uq_model"):
            try:
                uq_pred = _predict_component(self.uq_model, results)
                results["u_pred"] = np.asarray(uq_pred).ravel()
            except Exception:
                pass

        return results


class dissimilarity:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if hasattr(self, "pred") and callable(self.pred):
            try:
                return self.pred(X)
            except Exception:
                pass

        if hasattr(self, "model") and hasattr(self.model, "score_samples"):
            scores = self.model.score_samples(X)
            distances = -np.asarray(scores)
        elif hasattr(self, "model") and hasattr(self.model, "predict"):
            distances = np.asarray(self.model.predict(X))
        else:
            values = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
            distances = np.linalg.norm(values, axis=1)

        if hasattr(self, "scale"):
            try:
                distances = distances / float(self.scale)
            except Exception:
                pass

        return pd.DataFrame({"d_pred": np.asarray(distances).ravel()}, index=getattr(X, "index", None))


class calibration:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if hasattr(self, "model") and hasattr(self.model, "predict"):
            return self.model.predict(X)
        if hasattr(self, "uq_func") and callable(self.uq_func):
            return self.uq_func(X)
        return np.zeros(len(X))


class domain:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if hasattr(self, "model") and hasattr(self.model, "predict"):
            pred = self.model.predict(X)
        elif hasattr(self, "threshold") and "d_pred" in getattr(X, "columns", []):
            pred = np.asarray(X["d_pred"]) > float(self.threshold)
        elif "d_pred" in getattr(X, "columns", []):
            pred = np.asarray(X["d_pred"])
        else:
            pred = np.zeros(len(X))

        return pd.DataFrame({"domain_pred": np.asarray(pred).ravel()}, index=getattr(X, "index", None))


def _predict_component(component, X):
    if hasattr(component, "predict"):
        pred = component.predict(X)
    elif callable(component):
        pred = component(X)
    else:
        raise AttributeError(f"{type(component).__name__} cannot predict")

    if isinstance(pred, pd.DataFrame):
        if "d_pred" in pred.columns:
            return pred["d_pred"]
        if "y_pred" in pred.columns:
            return pred["y_pred"]
        return pred.iloc[:, 0]
    if isinstance(pred, pd.Series):
        return pred
    return pred
