
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_model.joblib"

EXPECTED_FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


def load_artifact(model_path=MODEL_PATH):
    """Load the saved model, threshold, and feature names."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    artifact = joblib.load(model_path)

    required_keys = {"model", "threshold", "features"}

    if not required_keys.issubset(artifact):
        raise ValueError("The model artifact has an invalid format.")

    if artifact["features"] != EXPECTED_FEATURES:
        raise ValueError("The saved feature schema is incorrect.")

    return artifact


def validate_features(data, feature_names=EXPECTED_FEATURES):
    """Validate and order transaction features."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [
        feature for feature in feature_names
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {', '.join(missing)}"
        )

    features = data.loc[:, feature_names].copy()

    for column in feature_names:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    values = features.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Features contain missing, non-numeric, or infinite values."
        )

    if (features["Amount"] < 0).any():
        raise ValueError("Transaction amounts cannot be negative.")

    if (features["Time"] < 0).any():
        raise ValueError("Transaction time cannot be negative.")

    return features


def predict_transactions(data, artifact=None):
    """Return fraud probabilities and threshold-based predictions."""
    if data.empty:
        raise ValueError("At least one transaction is required.")

    if artifact is None:
        artifact = load_artifact()

    features = validate_features(
        data,
        artifact["features"],
    )

    model = artifact["model"]
    threshold = float(artifact["threshold"])

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = data.copy()
    results["Fraud_Probability"] = probabilities
    results["Predicted_Class"] = predictions
    results["Prediction"] = np.where(
        predictions == 1,
        "Potential Fraud",
        "Not Flagged",
    )

    return results
