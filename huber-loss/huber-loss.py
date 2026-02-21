import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    e = y_true - y_pred
    return np.mean(np.where(np.abs(e) > delta, delta * (np.abs(e) - 1/2 * delta), 1/2 * e**2))
    pass