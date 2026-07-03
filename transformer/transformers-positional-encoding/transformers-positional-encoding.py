import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    position = np.arange(seq_length).reshape(-1, 1)
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    ans = np.zeros((seq_length, d_model))
    
    ans[:, 0::2] = np.sin(position * div)
    ans[:, 1::2] = np.cos(position * div)
    return ans
    