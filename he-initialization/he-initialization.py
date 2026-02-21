def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    L = math.sqrt(6 / fan_in)
    for i in range(len(W)):
        for j in range(len(W[0])):
            W[i][j] = W[i][j] * 2*L - L
    return W