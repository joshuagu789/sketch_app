import torch
from torch import nn
import matplotlib.pyplot as plt             # for seeing the images
import torch.nn.functional as F

from helpers import tensor_to_negative_one_to_one, tensor_to_zero_to_one
from gaussian_diffusion import guassian_diffusion

class trainer:
    """
    trains unet model, NOTE: dataset doesn't have images normalized so trainer does it here
    """

    def train_loop(self, noise_predictor, dataloader, batch_size, timesteps, learn_rate, epochs, device, optimizer, ):
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
        noiser = guassian_diffusion(num_timesteps=timesteps)

        for epoch in range(epochs):
            # for step, batch in enumerate(dataloader):
            for batch_number, img_list in enumerate(dataloader):   # img_list is list of 3d tensors for individual images
                print("Batch number " + str(batch_number+1) + ": ")
                for x_start in img_list:    # has dimensions [batch size, channels, width, height]

                    torch.cuda.empty_cache()    # experimental

                    # each img in batch has a random timesteps of noise applied to it (i think)- make neural network more experienced w/ different amounts of noise?  
                    noise = torch.randn_like(x_start)   

                    # plt.imshow(tensor_to_zero_to_one(x_start[5]).permute(1,2,0).cpu().detach().numpy())
                    # plt.imshow(tensor_to_zero_to_one(x_start_corrupted[5]).permute(1,2,0).cpu().detach().numpy())
                   
                    optimizer.zero_grad()

                    # first normalizes x_start, corrupts it with t = torch.randint(), then feeds it into neural network
                    predicted_noise = noise_predictor(noiser.q_sample(tensor_to_negative_one_to_one(x_start), torch.randint(0, timesteps, (batch_size,)), noise).to(device))    # note: hugging face says to implement timesteps into unet
                    
                    noise = noise.to(device)
                    loss = F.l1_loss(noise, predicted_noise)     # using l1 loss

                    loss.backward()
                    optimizer.step()
                    
                    print("Loss: " + str(loss.item()))


