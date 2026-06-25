import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.asarray(x, dtype=float)
    mean_x = np.mean(x, axis=0)
    n = x.shape[0]
    var = np.sum((x-mean_x)**2,axis=0)/(n-1)
    sd = np.sqrt(var)
    return var,sd