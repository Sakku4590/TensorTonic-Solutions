import numpy as np

def update_cell_state(C_prev: np.ndarray, f_t: np.ndarray,
                      i_t: np.ndarray, c_tilde: np.ndarray) -> np.ndarray:

    forgoten = f_t * C_prev
    input_info = i_t * c_tilde

    update_memory = forgoten + input_info

    return update_memory