import torch

"""
Contains useful methods, many follow style of helpers.py in minimagen 
https://github.com/AssemblyAI-Examples/MinImagen/blob/0f305c29922274e1faefe9e93be441fdb7ed0efa/minimagen/helpers.py
"""

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    COPY PASTED FROM LINK AT TOP

    Extracts values from `a` using `t` as indices

    :param a: 1D tensor of length L.
    :param t: 1D tensor of length b.
    :param x_shape: Tensor of size (b, c, h, w).
    :return: Tensor of shape (b, 1, 1, 1) that selects elements of a, using t as indices of selection.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))