import numpy as np

def get_opt(Q):
    return np.argmax(Q, axis=-1)