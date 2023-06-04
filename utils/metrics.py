import torch

def compute_accuracy(y_true, y_hat):
    """
    Computes the accuracy of the predictions y_hat against the true labels y_true

    Args:
        y_true (torch.Tensor): true labels
        y_hat (torch.Tensor): predicted labels

    Returns:
        float: accuracy

    Examples:
        >>> compute_accuracy(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
        1.00
        >>> compute_accuracy(torch.tensor([0, 1, 2]), torch.tensor([2, 1, 0]))
        0.00
        >>> compute_accuracy(torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]))
        0.33
    """
    assert len(y_true) == len(y_hat), "y_true and y_hat must have the same length"
    assert len(y_true) > 0, "y_true and y_hat must have at least one element"

    return torch.mean((y_true == y_hat).float()).item()
    