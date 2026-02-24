import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.array([])
    if max_len == None:
        max_len = max(len(seq) for seq in seqs)
    matris = np.full((len(seqs),max_len),pad_value)

    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            if j >= max_len:
                pass
            else:
                matris[i,j]= seqs[i][j]
    return matris
    pass