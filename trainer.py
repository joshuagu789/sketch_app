import torch
from torch import nn
from torch.utils.data import DataLoader     # transforms data and labels into iterables so that life easier
from torchvision import datasets            # stores data and their labels (also contains fashion mnist data set)
from torchvision.transforms import ToTensor # converts python pil image or n dimensional array into tensor of choice
import matplotlib.pyplot as plt             # for seeing the images
from torch.optim import Adam                # optimizer
import torch.nn.functional as F

class trainer:
    """
    kaggle datasets download -d iamsouravbanerjee/animal-image-dataset-90-different-animals
    """

    device = "cuda" if torch.cuda.is_available else "cpu"
    print(f"using {device} device")
    # model = neural_network().to(device)
    # print(model.parameters)

    learning_rate = 0.001
    epochs = 5
    loss_function = nn.CrossEntropyLoss()  

    def train_loop(noise_predictor, dataloader, t, learn_rate, epochs):
        """
        Takes in the unet model and a dataset for training noise prediction

        noise_predictor: the unet model created from u_net_no_conditioning.py
        dataloader: dataset converted into pytorch Dataloader
        t: for timesteps
        learn_rate: how fast weights are adjusted
        epochs: number of iterations of training
        """
        optimizer = Adam(noise_predictor.parameters(), lr=learn_rate)

        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                noise = torch.randn_like(batch)

                predicted_noise = noise_predictor(batch)    # note: hugging face says to implement timesteps into unet
                loss = F.l1_loss(noise, predicted_noise)     # using l1 loss

                if step % 100 == 0:
                    print("Loss: " + str(loss.item()))

                loss.backward()
                optimizer.step()
                
                return