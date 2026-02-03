import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    
    # step 1 =  concatenate
    c = np.concatenate((h_prev,x_t), axis = 1)

    # step 2 = linier transform
    z = c @ W_f.T + b_f

    # step 3 = apply sigmoid 
    x = sigmoid(z)
    return x