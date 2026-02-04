import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    
    # step 1 = concate
    concate = np.concatenate((h_prev,x_t), axis = 1)

    # forgate gate
    f_t = sigmoid((concate @ W_f.T)+ b_f)

    # input gate
    i_t = sigmoid((concate @ W_i.T)+b_i)

    # candidate memory
    ct = np.tanh((concate @ W_c.T)+b_c)

    # Memory
    c_t = f_t * C_prev + i_t * ct

    # hiden state
    h_t = b_o * np.tanh(c_t)

    return h_t, c_t
