def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    pt = predictions
    for i in range(len(predictions)):
        if targets[i] == 0:
            pt[i] = -1 * alpha * (1- (1- predictions[i]))**gamma * math.log(1-predictions[i])
        elif targets[i] == 1:
            pt[i] = -1 * alpha * (1- predictions[i])**gamma * math.log(predictions[i])
    summ = 0
    for i in pt:
        summ+= i
    return summ/len(targets)