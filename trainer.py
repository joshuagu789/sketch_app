import torch
from torch import nn
from torch.utils.data import DataLoader     # transforms data and labels into iterables so that life easier
import matplotlib.pyplot as plt             # for seeing the images
from torch.optim import Adam                # optimizer
import torch.nn.functional as F
from torchvision import transforms
from helpers import tensor_to_negative_one_to_one, tensor_to_zero_to_one

class trainer:
    """
    trains unet model, NOTE: dataset doesn't have images normalized so trainer does it here
    """

    device = "cuda" if torch.cuda.is_available else "cpu"
    print(f"using {device} device")
    # model = neural_network().to(device)
    # print(model.parameters)

    learning_rate = 0.001
    epochs = 5
    loss_function = nn.CrossEntropyLoss()  

    def train_loop(self, noise_predictor, dataloader, batch_size, timesteps, learn_rate, epochs, device):
        """
        Takes in the unet model and a dataset for training noise prediction

        noise_predictor: the unet model created from u_net_no_conditioning.py
        dataloader: dataset converted into pytorch Dataloader
        batch_size:
        timesteps:
        learn_rate: how fast weights are adjusted
        epochs: number of iterations of training
        device: CUDAAA
        """
        optimizer = Adam(noise_predictor.parameters(), lr=learn_rate)

        for epoch in range(epochs):
            # for step, batch in enumerate(dataloader):
            for batch, img_list in enumerate(dataloader):   # img_list is list of 3d tensors for individual images
                print("Batch number " + str(batch+1) + ": ")
                for x_start in img_list:    # has dimensions [batch size, channels, width, height]

                    x_start = tensor_to_negative_one_to_one(x_start)
                    t = torch.randint(0, timesteps, (batch_size,))  

                    # plt.imshow(tensor_to_zero_to_one(x_start[7]).permute(1,2,0).cpu().detach().numpy())
                    optimizer.zero_grad()

                    predicted_noise = noise_predictor(x_start.to(device))    # note: hugging face says to implement timesteps into unet
                    
                    noise = torch.randn_like(predicted_noise)
                    noise = noise.to(device)

                    loss = F.l1_loss(noise, predicted_noise)     # using l1 loss

                    loss.backward()
                    optimizer.step()
                    
                    print("Loss: " + str(loss.item()))

