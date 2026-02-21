import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    if x.ndim==4:
        gamma = np.array(gamma)
        beta = np.array(beta)
        axis = (0, 2, 3)
        gamma = gamma.reshape(1,-1,1,1)
        beta = beta.reshape(1,-1,1,1)
    elif x.ndim==2:
        axis = 0
    m = np.mean(x,axis=axis, keepdims=True)
    var = np.var(x,axis = axis, keepdims=True)
    xi = (x-m)/np.sqrt(var + eps)
    return gamma * xi + beta
    pass