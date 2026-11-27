import numpy as np
import pytest

from src.evaluation.metrics import evaluate_predictions
from src.evaluation.threshold import select_threshold


@pytest.fixture
def config():
    return {
        "validation": {
            "threshold_selection": {
                "metric": "f2",
                "beta": 2,
                "search": {
                    "start": 0.01,
                    "stop": 0.99,
                    "step": 0.01,
                },
                "tie_break": "higher_threshold",
            }
        }
    }


def test_evaluate_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 0]
    y_proba = [0.1, 0.2, 0.9, 0.4]

    results = evaluate_predictions(y_true, y_pred, y_proba)

    assert results["precision"] == 1.0
    assert results["recall"] == 0.5
    assert results["f1"] == pytest.approx(2 / 3)
    assert results["f2"] == pytest.approx(5 / 9)
    assert results["confusion_matrix"] == [[2, 0], [1, 1]]


def test_evaluate_predictions_rejects_misaligned_data():
    with pytest.raises(ValueError):
        evaluate_predictions([0, 1], [0], [0.1, 0.9])


def test_evaluate_predictions_rejects_single_class():
    with pytest.raises(ValueError):
        evaluate_predictions([0, 0], [0, 0], [0.1, 0.2])


def test_select_threshold(config):
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    result = select_threshold(y_true, y_proba, config)

    assert result["threshold"] == pytest.approx(0.8)
    assert result["f_beta"] == 1.0
    assert result["beta"] == 2.0


def test_select_threshold_rejects_misaligned_data(config):
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0.1], config)


def test_select_threshold_rejects_invalid_probabilities(config):
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0.1, 1.5], config)


def test_select_threshold_rejects_single_class(config):
    with pytest.raises(ValueError):
        select_threshold([0, 0], [0.1, 0.2], config)

def test_select_threshold_tie_breaks_with_higher_threshold(config):
    y_true = np.array([0, 1])
    y_proba = np.array([0.5, 0.5])

    result = select_threshold(y_true, y_proba, config)

    assert result["threshold"] == pytest.approx(0.50)