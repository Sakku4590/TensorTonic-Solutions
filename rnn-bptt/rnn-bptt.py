import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    # YOUR CODE HERE
    dz = dh_next * (1 - h_t ** 2)

    dh_prev = np.dot(dz, W_hh)

    dW_hh = np.dot(dz.T, h_prev)

    return dh_prev, dW_hh