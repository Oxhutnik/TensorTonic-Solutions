import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(seq_len)[:, np.newaxis] # Shape: (seq_len, 1)
    
    # 2. Senin 'row' değişkenin (frekanslar)
    # 1+d_model//2 yerine tam d_model/2 kadar frekans yeterli
    row_i = np.arange(0, d_model, 2) 
    row = 1 / (base ** (row_i / d_model)) # Shape: (d_model/2,)

    # 3. PE matrisini oluştur (ones yerine zeros daha güvenli)
    pe = np.zeros((seq_len, d_model))

    # 4. Senin atama mantığın (Broadcast kullanarak)
    # pos (seq_len, 1) * row (d_model/2) -> (seq_len, d_model/2) matrisi üretir
    pe[:, 0::2] = np.sin(pos * row)
    
    # int kontrolü yerine modülo kullanıyoruz
    if d_model % 2 == 0:
        pe[:, 1::2] = np.cos(pos * row)
    else:
        # Tek sayı d_model durumu için son sütunu dışarıda bırakıyoruz
        pe[:, 1::2] = np.cos(pos * row[:-1])
        
    return pe