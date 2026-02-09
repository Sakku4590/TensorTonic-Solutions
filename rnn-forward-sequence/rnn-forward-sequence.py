import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    
    batch_size,time_step, _ = X.shape

    hidden_dim = h_0.shape[1]

    h_all = np.zeros((batch_size,time_step,hidden_dim))

    # Initialize hidden state
    h_t = h_0

    # Unrole RNN over time

    for t in range (time_step):
      x_t = X[:, t, :]

      h_t = np.tanh(
          np.dot(x_t, W_xh.T) +
          np.dot(h_t, W_hh.T) +
          b_h
      )

      h_all[:, t, :] = h_t

    h_final = h_t
    return h_all, h_final
