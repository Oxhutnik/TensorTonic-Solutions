import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.array(x)
    if x.ndim == 4:
        return np.mean(x, axis= (2,3))
    elif x.ndim == 3:
        return np.mean(x, axis = (1,2))
    else:
        raise ValueError
    pass