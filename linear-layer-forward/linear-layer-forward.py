def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    X = np.array(X, dtype=float)
    W = np.array(W, dtype=float)
    b = np.array(b, dtype=float)

    # Compute linear transformation
    Y = X @ W + b   # same as np.dot(X, W) + b

    return Y.tolist()