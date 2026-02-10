import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    scale = np.linalg.norm(W_hh, 2)

    grads = [1.0]

    for _ in range(1, T):
        grads.append(grads[-1] * scale)
    
    return grads