import torch
from torch import nn
from torch.utils.data import DataLoader     # transforms data and labels into iterables so that life easier
from torchvision import datasets            # stores data and their labels (also contains fashion mnist data set)
from torchvision.transforms import ToTensor # converts python pil image or n dimensional array into tensor of choice
import matplotlib.pyplot as plt             # for seeing the images

class trainer:
    """
    
    """

    device = "cuda" if torch.cuda.is_available else "cpu"
    print(f"using {device} device")
    # model = neural_network().to(device)
    # print(model.parameters)

    learning_rate = 0.001
    epochs = 5
    loss_function = nn.CrossEntropyLoss()  

    def train_loop():
       return