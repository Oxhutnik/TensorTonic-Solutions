import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    aps = []
    
    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.array(y_true)
        y_score = np.array(y_score)

        order = np.argsort(-y_score)
        if k is not None:
            order = order[:k]
        
        y_true_sorted = y_true[order]
        
        hits = np.cumsum(y_true_sorted) 

        precision_at_k = hits / np.arange(1, len(y_true_sorted) + 1)

        relevant_precisions = precision_at_k * y_true_sorted
        
        R = np.sum(y_true)
        
        if R > 0:
            ap = np.sum(relevant_precisions) / R
            aps.append(ap)
        else:
            aps.append(0.0)

    mAP = np.mean(aps)
    return (mAP,aps)
    
    
    pass