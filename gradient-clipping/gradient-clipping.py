import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.array(g, dtype=float)

    # Compute L2 norm
    norm = np.sqrt(np.sum(g ** 2))

    # If max_norm <= 0 → return gradient unchanged
    if max_norm <= 0:
        return g

    # Scale only if norm exceeds threshold
    if norm > max_norm and norm > 0:
        g = g * (max_norm / norm)

    return g