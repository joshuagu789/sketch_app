import torch
from torch import nn
import matplotlib.pyplot as plt             # for seeing the images
import torch.nn.functional as F

from helpers import tensor_to_negative_one_to_one, tensor_to_zero_to_one
from gaussian_diffusion import guassian_diffusion
from guassian_diffusion_copy_pasta import forward_diffusion_sample

class trainer:
    """
    trains unet model, NOTE: dataset doesn't have images normalized so trainer does it here
    """

    def train_loop(self, noise_predictor, dataloader, batch_size, timesteps, epochs, device, optimizer, scheduler, epoch_history):
        """
        Takes in the unet model and a dataset for training noise prediction

        noise_predictor: the unet model created from u_net_no_conditioning.py
        dataloader: dataset converted into pytorch Dataloader
        batch_size:
        timesteps:
        learn_rate: how fast weights are adjusted
        epochs: number of iterations of training
        device: CUDAAA
        optimizer: contains learn rate
        epoch_history: contains loss across each epoch- useful if model already had prior training on x epoch's
        """
        noiser = guassian_diffusion(num_timesteps=timesteps)
        epochs_done = len(epoch_history)
        # for name, param in noise_predictor.named_parameters():
        #             print(name, param.grad, param.requires_grad)
        for epoch in range(epochs):

            loss_sum = 0
            number_batches = 0

            # for step, batch in enumerate(dataloader):
            for batch_number, img_list in enumerate(dataloader):   # img_list is list of 3d tensors for individual images
                # print("Batch number " + str(batch_number+1) + ": ")
                for x_start in img_list:    # has dimensions [batch size, channels, width, height]

                    # torch.cuda.empty_cache()    # experimental

                    # each img in batch has a random timesteps of noise applied to it (i think)- make neural network more experienced w/ different amounts of noise?  
                    # noise = torch.randn_like(x_start)   
                    # t = torch.randint(0, timesteps, (batch_size,))
                    t = torch.randint(0, timesteps, (x_start.size(dim=0),))
                    # plt.imshow((x_start[5]).permute(1,2,0).cpu().detach().numpy())
                    # plt.imshow(tensor_to_zero_to_one(noiser.q_sample(tensor_to_negative_one_to_one(x_start), torch.randint(0, timesteps, (batch_size,)), noise)[5]).permute(1,2,0).cpu().detach().numpy())
                   
                    optimizer.zero_grad()

                    # first normalizes x_start, corrupts it with t = torch.randint(), then feeds it into neural network
#predicted_noise = noise_predictor(noiser.q_sample(tensor_to_negative_one_to_one(x_start), t, noise).to(device), t.to(device))    # note: hugging face says to implement timesteps into unet
                    # print("max/min: " + str(torch.max(predicted_noise)) + "/" + str(torch.min(predicted_noise)))
                    # plt.imshow(tensor_to_zero_to_one(predicted_noise[5]).permute(1,2,0).cpu().detach().numpy())

                    noisy_img, noise = forward_diffusion_sample(tensor_to_negative_one_to_one(x_start), t, device)
                    # plt.imshow(tensor_to_zero_to_one(noisy_img[9]).permute(1,2,0).cpu().detach().numpy())
                    predicted_noise = noise_predictor(noisy_img.to(device), t.to(device))
                    loss = F.l1_loss(noise, predicted_noise)     # using l1 loss

                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                    loss_sum += loss.item()
                    number_batches += 1
                    # print("Loss: " + str(loss.item()))
            
            average_loss = float(loss_sum/number_batches)
            print("Loss in epoch " + str(epoch + 1 + epochs_done) +": " + str(average_loss))
            epoch_history.append(round(average_loss,3))

            # print("active: " + torch.cuda.get_device_name(0) + ", core count: " + str(torch.cuda.device_count()))

        print("Loss over all epochs: " + str(epoch_history))
        plt.plot(epoch_history)
        # for name, param in noise_predictor.named_parameters():
        #             print(name, param.requires_grad)
                    # print(name, param.grad, param.requires_grad)


