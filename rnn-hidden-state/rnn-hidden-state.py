import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:

    """
    Initialize the hidden state for an RNN.

    Parameters:
    batch_size : int
        Number of sequences in the batch
    hidden_dim : int
        Size of the hidden state

    Returns:
    h_0 : ndarray of shape (batch_size, hidden_dim)
        Initialized hidden state (zeros)
    """
    return np.zeros((batch_size, hidden_dim))