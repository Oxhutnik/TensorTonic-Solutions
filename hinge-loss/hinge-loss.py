import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    y_score = np.array(y_score)
    y_true = np.array(y_true)
    e = y_score * y_true
    if reduction == "mean":
        return np.mean(np.where((margin - e)>0, margin - e, 0))
    elif reduction == "sum":
        return np.sum(np.where((margin - e)>0, margin - e, 0))
