def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    pr = predictions[:]
    for i in range(len(predictions)):
        if i == target:
            predictions[i] = (1-epsilon) + epsilon/len(predictions)
        else:
            predictions[i] = epsilon/len(predictions)
    summ = 0
    for i in range(len(predictions)):
        summ += predictions[i] * math.log(pr[i])
    return -summ