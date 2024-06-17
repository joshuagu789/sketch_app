import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import from_numpy

def cartoonify(tensor):
    """
    Takes tensor of dimensions (C, W, H) with values zero to one and 
    returns a tensor of same dimensions with values zero to one but image is cartoon styled
    """
    tensor_numpy = tensor.cpu().detach().numpy()
    tensor_numpy = (np.moveaxis(tensor_numpy, 0, -1) * 255).astype(np.uint8)

    hsv_img = cv2.cvtColor(tensor_numpy, cv2.COLOR_BGR2HSV)
    hsv_img[...,1] = hsv_img[...,1] * 1.35
    hsv_img[...,2] = (hsv_img[...,2]) * 1.25

    # tensor_numpy = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    # plt.imshow(tensor_numpy)  

    for x in range(len(hsv_img)):
        for y in range(len(hsv_img[x])):
            for z in range(len(hsv_img[x][y])):
                if hsv_img[x][y][z] > 255:
                   hsv_img[x][y][z] = 255 
    tensor_numpy = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    # plt.imshow(tensor_numpy)  

    edges = cv2.Canny(tensor_numpy, 50, 150)   # originally 300-350 and 100-200
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # plt.imshow(edges)

    avg_value = [0,0,0]
    counter = 0
    for x in range(len(edges)):
        for y in range(len(edges[x])):
            for z in range(3):
                avg_value[z] += tensor_numpy[x][y][z]
            counter += 1
    avg_value[0] = avg_value[0] / counter
    avg_value[1] = avg_value[1] / counter
    avg_value[2] = avg_value[2] / counter

    for x in range(len(edges)):
        for y in range(len(edges[x])):
            if edges[x][y][0] == 255: 
                for z in range(3):
                    tensor_numpy[x][y][z] = avg_value[z]
    # plt.imshow(tensor_numpy)

    return from_numpy(tensor_numpy)


