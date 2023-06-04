import math
import torch
from utils.metrics import compute_accuracy

def test_metrics():
    # 1. Test for equal predictions and true labels
    y_true = torch.tensor([0, 1, 2])
    y_hat = torch.tensor([0, 1, 2])
    expected_accuracy = 1.0

    assert compute_accuracy(y_true, y_hat) == expected_accuracy

    # 2. Test for completely incorrect predictions
    y_true = torch.tensor([0, 1, 2])
    y_hat = torch.tensor([2, 0, 1])
    expected_accuracy = 0.0

    assert compute_accuracy(y_true, y_hat) == expected_accuracy

    # 3. Test for partially correct predictions
    y_true = torch.tensor([0, 1, 2])
    y_hat = torch.tensor([0, 0, 0])
    expected_accuracy = 1/3

    computed_accuracy = compute_accuracy(y_true, y_hat)
    assert math.isclose(computed_accuracy, expected_accuracy, rel_tol=1e-5)

    # 4. Test for empty input tensors
    y_true = torch.tensor([])
    y_hat = torch.tensor([])

    # Check that raises an AssertionError
    try:
        compute_accuracy(y_true, y_hat)
        assert False
    except AssertionError:
        assert True

    # 5. Check different lengths
    y_true = torch.tensor([0, 1, 2])
    y_hat = torch.tensor([0, 1, 2, 3])

    # Check that raises an AssertionError
    try:
        compute_accuracy(y_true, y_hat)
        assert False
    except AssertionError:
        assert True
