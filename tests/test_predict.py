import numpy as np
import pandas as pd
import pytest

from src.predict import (
    EXPECTED_FEATURES,
    predict_transactions,
    validate_features,
)


@pytest.fixture
def sample_transactions():
    """Create two valid sample transactions."""
    return pd.DataFrame(
        [
            [100.0] + [0.1] * 28 + [50.0],
            [200.0] + [0.2] * 28 + [100.0],
        ],
        columns=EXPECTED_FEATURES,
    )


class DummyModel:
    """A predictable model for testing without a trained model."""

    def predict_proba(self, features):
        return np.array([
            [0.9, 0.1],
            [0.2, 0.8],
        ])[:len(features)]


@pytest.fixture
def dummy_artifact():
    return {
        "model": DummyModel(),
        "threshold": 0.5,
        "features": EXPECTED_FEATURES,
    }


def test_valid_features(sample_transactions):
    result = validate_features(sample_transactions)

    assert list(result.columns) == EXPECTED_FEATURES
    assert result.shape == (2, 30)


def test_feature_order_is_correct(sample_transactions):
    shuffled = sample_transactions[
        list(reversed(EXPECTED_FEATURES))
    ]

    result = validate_features(shuffled)

    assert list(result.columns) == EXPECTED_FEATURES


def test_missing_feature_raises_error(sample_transactions):
    incomplete = sample_transactions.drop(columns=["V1"])

    with pytest.raises(ValueError, match="Missing required features"):
        validate_features(incomplete)


def test_non_numeric_feature_raises_error(sample_transactions):
    invalid = sample_transactions.copy().astype(object)
    invalid.loc[0, "V1"] = "invalid"

    with pytest.raises(ValueError, match="missing, non-numeric"):
        validate_features(invalid)


def test_negative_amount_raises_error(sample_transactions):
    invalid = sample_transactions.copy()
    invalid.loc[0, "Amount"] = -10

    with pytest.raises(ValueError, match="amounts cannot be negative"):
        validate_features(invalid)


def test_predictions_have_expected_columns(
    sample_transactions, dummy_artifact
):
    result = predict_transactions(
        sample_transactions, dummy_artifact
    )

    assert "Fraud_Probability" in result.columns
    assert "Predicted_Class" in result.columns
    assert "Prediction" in result.columns
    assert len(result) == 2


def test_threshold_classification(
    sample_transactions, dummy_artifact
):
    result = predict_transactions(
        sample_transactions, dummy_artifact
    )

    assert result["Predicted_Class"].tolist() == [0, 1]
    assert result["Prediction"].tolist() == [
        "Not Flagged",
        "Potential Fraud",
    ]


def test_empty_input_raises_error(dummy_artifact):
    empty_df = pd.DataFrame(columns=EXPECTED_FEATURES)

    with pytest.raises(ValueError, match="At least one transaction"):
        predict_transactions(empty_df, dummy_artifact)