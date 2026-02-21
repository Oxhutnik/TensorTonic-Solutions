import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g)
    norm_g = np.linalg.norm(g)
    if norm_g <= max_norm or max_norm <=0 or norm_g == 0:
        return g.copy()
    else:
        return g * (max_norm/norm_g)
    pass