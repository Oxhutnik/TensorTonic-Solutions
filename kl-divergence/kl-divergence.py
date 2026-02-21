import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    q = np.array(q)
    p = np.array(p)

    return np.sum(p * np.log(p/q + eps))