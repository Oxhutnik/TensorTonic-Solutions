import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    s = np.array(s)
    w = np.array(w)
    g = np.array(g)
    s = beta*s + (1-beta)*(g*g)
    w = w - (lr/(np.sqrt(s+eps)) * g)
    return (w,s)
    pass
    