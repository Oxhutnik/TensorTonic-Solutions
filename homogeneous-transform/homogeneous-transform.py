import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """

    T = np.array(T)
    p = np.array(points)
    if p.ndim ==1:
        p = p.reshape(-1,1)
        ph = np.ones((p.shape[0]+1,p.shape[1]))
        ph[:3] = p
    
        p_new= T@ph
        return  p_new[:3,:].flatten()
    elif p.ndim==2:
        ph = np.ones((p.shape[0],p.shape[1]+1))
        ph[:,:3] = p
    
        p_new= ph@T.T
        return  p_new[:,:3]