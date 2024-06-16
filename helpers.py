import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

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

def tensor_to_negative_one_to_one(tensor: torch.Tensor):
    """
    normalizes 4D tensor representing batch of images
    """
    return tensor * 2 - 1

def tensor_to_zero_to_one(tensor: torch.Tensor):
    """
    unormalizes 4D tensor representing batch of images
    """
    return (tensor + 1) * 0.5 

# COPY PASTA FROM https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=k13hj2mciCHA
def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def show_tensor_image(image):
    """
    copy pastad from https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=k13hj2mciCHA
    """
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))