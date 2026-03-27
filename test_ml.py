import pytest
# TODO: add necessary import
import numpy as np
from ml.model import train_model, compute_model_metrics, inference

X = np.array([[1, 0], [2, 0], [0, 1], [0, 2]], dtype=float)
y = np.array([0, 0, 1, 1])

# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_model():
    """Train model returns an object with a predict method."""
    model = train_model(X, y)
    assert hasattr(model, "predict")


def test_inference_output_shape():
    """Inference returns an array with one prediction per input sample."""
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == (X.shape[0],)


def test_compute_metrics_perfect_predictions():
    """Perfect predictions should yield precision, recall, and F1 of 1.0."""
    precision, recall, fbeta = compute_model_metrics(y, y)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0