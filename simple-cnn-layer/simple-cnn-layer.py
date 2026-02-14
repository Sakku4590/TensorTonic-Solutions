import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    x = np.array(x, dtype=float)
    W = np.array(W, dtype=float)
    b = np.array(b, dtype=float)

    N, C_in, H, W_in = x.shape
    C_out, C_in_w, KH, KW = W.shape

    # Safety check
    if C_in != C_in_w:
        raise ValueError("Input channels must match weight channels")

    # Output dimensions
    H_out = H - KH + 1
    W_out = W_in - KW + 1

    # Initialize output tensor
    y = np.zeros((N, C_out, H_out, W_out))

    # Perform convolution
    for n in range(N):
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):

                    total = 0.0

                    for c_in in range(C_in):
                        for u in range(KH):
                            for v in range(KW):
                                total += (
                                    x[n, c_in, i + u, j + v] *
                                    W[c_out, c_in, u, v]
                                )

                    y[n, c_out, i, j] = total + b[c_out]

    return y
  
    
  