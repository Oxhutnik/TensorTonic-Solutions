def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    L = math.sqrt(6 / (fan_in+fan_out))
    for i in range(len(W)):
        for j in range(len(W[0])):
            W[i][j] = W[i][j] * 2*L - L
    return W