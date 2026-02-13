import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.array(x, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)

    if x.ndim == 2:
        # Shape: (N, D)
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_hat + beta

    elif x.ndim == 4:
        # Shape: (N, C, H, W)
        mean = np.mean(x, axis=(0,2,3), keepdims=True)
        var = np.var(x, axis=(0,2,3), keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        # reshape gamma and beta for broadcasting
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

        out = gamma * x_hat + beta

    else:
        raise ValueError("Input must be 2D or 4D")

    return out.tolist()