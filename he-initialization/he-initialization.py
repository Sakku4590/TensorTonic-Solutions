def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    W = np.array(W, dtype= float)
    # fan_in = np.array(fan_in, dtype= float)

    L = np.sqrt(6 / fan_in)

    z = W * (2*L) - L

    return z.tolist()