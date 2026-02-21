import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(seq_len)[:, np.newaxis]

    row_i = np.arange(0, d_model, 2) 
    row = 1 / (base ** (row_i / d_model)) 

    pe = np.zeros((seq_len, d_model))

    pe[:, 0::2] = np.sin(pos * row)
    
    if d_model % 2 == 0:
        pe[:, 1::2] = np.cos(pos * row)
    else:
        pe[:, 1::2] = np.cos(pos * row[:-1])
        
    return pe