import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    real_scores_mean = np.mean(np.array(real_scores))
    fake_scores_mean = np.mean(np.array(fake_scores))
    
    return fake_scores_mean - real_scores_mean
    pass