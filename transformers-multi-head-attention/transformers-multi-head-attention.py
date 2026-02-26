import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    B = Q.shape[0]
    seq_len = Q.shape[1]
    dk = int(Q.shape[2]/num_heads)
    final_heads = np.ones((B,seq_len,Q.shape[2]))
    for i in range(B):
        head_outputs = []
        for j in range(num_heads):
            q_p = Q[i] @ W_q[:, j*dk : (j+1)*dk]
            k_p = K[i] @ W_k[:, j*dk : (j+1)*dk]
            v_p = V[i] @ W_v[:, j*dk : (j+1)*dk]
            
            attn = softmax((q_p @ k_p.T) / np.sqrt(dk))
            head_out = attn @ v_p 
            head_outputs.append(head_out)
        
        combined_heads = np.concatenate(head_outputs, axis=-1) 
        
        final_heads[i] = combined_heads @ W_o
    
    return final_heads
    pass