import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    
    concate = np.concatenate((h_prev,x_t), axis = 1)
    
    y_i = concate @ W_i.T + b_i
    y_c = concate @ W_c.T + b_c

    x_i = sigmoid(y_i)
    x_c = np.tanh(y_c)
    return x_i,x_c