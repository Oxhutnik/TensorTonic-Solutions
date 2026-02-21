def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    total = 0
    norm_b = 0
    norm_a = 0

    for a,b in zip(x1,x2):
        total += a * b
        norm_a += a**2
        norm_b += b**2
    norma = math.sqrt(norm_a)
    normb = math.sqrt(norm_b)
    if label ==1:
        return 1- total/(norma * normb)
    if label ==  -1:
        return total/(norma * normb) - margin if (total/(norma * normb) - margin) > 0 else 0