import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.asarray(x,dtype=np.float64)

    positive = x >= 0
    negative = ~positive

    out = np.empty_like(x, dtype=np.float64)

    out[positive] = 1/(1 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    out[negative] = exp_x/(1+exp_x)

    return out