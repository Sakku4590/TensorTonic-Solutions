import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    
    f_g = np.dot(h_prev,W_hh.T)

    s_g = np.dot(x_t, W_xh.T)

    z = np.tanh(f_g + s_g + b_h)

    return z
