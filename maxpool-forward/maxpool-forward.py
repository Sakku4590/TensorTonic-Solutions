def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    H = len(X)
    W = len(X[0])

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    output = [[0 for _ in range(W_out)] for _ in range(H_out)]

    for i in range(H_out):
        for j in range(W_out):
            max_val = float('-inf')
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[i * stride + a][j * stride + b]
                    if val > max_val:
                        max_val = val
            output[i][j] = max_val

    return output






