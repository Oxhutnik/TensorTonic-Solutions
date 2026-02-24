import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    aps = []
    
    # Her bir sorguyu (query) tek tek geziyoruz
    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.array(y_true)
        y_score = np.array(y_score)

        # 1. Skorlara göre büyükten küçüğe dizen indeksleri bul
        order = np.argsort(-y_score)
        if k is not None:
            order = order[:k]
        
        # 2. Etiketleri bu sıraya göre diz (Doğrular/Yanlışlar sıralandı)
        y_true_sorted = y_true[order]
        
        # 3. Cumsum ile "o ana kadar kaç doğru var"ı bul
        hits = np.cumsum(y_true_sorted) # [1, 2, 2, 3] gibi
        
        # 4. "O ana kadarki toplam eleman sayısı"na böl (Hassasiyet/Precision)
        # [1/1, 2/2, 2/3, 3/4] gibi
        precision_at_k = hits / np.arange(1, len(y_true_sorted) + 1)
        
        # 5. KRİTİK NOKTA: Sadece gerçekten DOĞRU olan (1 olan) yerlerdeki 
        # hassasiyetleri topla ve toplam doğru sayısına (R) böl.
        relevant_precisions = precision_at_k * y_true_sorted
        
        # Toplam doğru sayısı (R)
        R = np.sum(y_true)
        
        if R > 0:
            ap = np.sum(relevant_precisions) / R
            aps.append(ap)
        else:
            aps.append(0.0) # Hiç doğru sonuç yoksa AP 0'dır
            
    # Tüm sorguların AP ortalamasını al = mAP
    mAP = np.mean(aps)
    return (mAP,aps)
    
    
    pass