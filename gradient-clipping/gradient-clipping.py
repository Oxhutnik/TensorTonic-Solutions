import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
# İpucu 1: Girdiyi np.ndarray formatına çevir (zaten array ise gereksiz kopya oluşturmaz)
    g = np.asarray(g)
    
    # İpucu 3 & Gereksinimler: Negatif veya sıfır max_norm durumunda orijinal matrisi döndür
    if max_norm <= 0:
        return g.copy()
        
    # İpucu 1: Tüm gradyan matrisinin L2 normunu hesapla
    norm_g = np.linalg.norm(g)
    
    # İpucu 2 & 3: Norm sıfırsa veya zaten sınırın (max_norm) altındaysa kopyasını döndür
    if norm_g == 0 or norm_g <= max_norm:
        return g.copy()
        
    # İpucu 3: Sınırı aşıyorsa yönü koruyarak büyüklüğü ölçeklendir (scale)
    return g * (max_norm / norm_g)