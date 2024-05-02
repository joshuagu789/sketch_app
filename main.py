from PIL import Image
from diffusers.utils import load_image
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim import Adam                # optimizer

from torch.utils.data import DataLoader, TensorDataset
import opendatasets as od
import os
import glob

from gaussian_diffusion import guassian_diffusion
from u_net_no_conditioning import u_net_no_conditioning
from trainer import trainer
from image_loader import image_loader

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    timesteps = 1000
    batch_size = 10
    learn_rate = 0.1
    epochs = 1
    counter = 0
    rawdataset = []   # for process of converting jpg images into 4d tensor

    image_loader = image_loader()
    trainer = trainer()
    pil_to_tensor = transforms.ToTensor()

    # state = torch.load("u_net01_and_optimizer.pth")     # contains state_dict and optimizer
    # u_net = u_net_no_conditioning()
    # u_net.load_state_dict(state["state_dict"])
    # u_net = u_net.to(device)
    # optimizer = Adam(u_net.parameters(), lr=learn_rate)  
    # optimizer.load_state_dict(state["optimizer"]) 

    # tensor_data = torch.load("ninety-animals.pt") 
    # tensor_dataset = TensorDataset(tensor_data)
    # dataloader = DataLoader(tensor_dataset, batch_size = batch_size, shuffle = True, pin_memory=True)

    # trainer.train_loop(u_net, dataloader, batch_size, timesteps, learn_rate, epochs, device, optimizer)

    # state = {
    #     "state_dict": u_net.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    # }
    # torch.save(state, "u_net01_and_optimizer.pth")
    
    """
    Loading raw jpg images and saving them as 4d tensor
    """
    # od.download(
    #     "https://www.kaggle.com/datasets/gpiosenka/100-bird-species"
    # )    
    # od.download(
    #     "https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals"
    # )

    tensordataset = image_loader.store_images_from_directory_as_tensor("100-bird-species/train/HARPY EAGLE")
    print(tensordataset.size())
    # plt.imshow(tensordataset[5].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[53].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[97].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[167].permute(1,2,0).cpu().detach().numpy())
    torch.save(tensordataset, "harpy_eagles01.pt")
    # torch.save(tensordataset, "ninety-animals.pt")
    
    # tensordataset = torch.load("ninety-animals.pt")

    # plt.imshow(tensordataset[541].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[1331].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[3911].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[4912].permute(1,2,0).cpu().detach().numpy())

    # plt.imshow(tensordataset[713].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[2412].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[4234].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[5369].permute(1,2,0).cpu().detach().numpy())

    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )

    image_as_tensor = pil_to_tensor(image)
    image_as_tensor = image_as_tensor[None,:,:,:]
    image_as_tensor = image_as_tensor.to(torch.float32)

    # t = torch.linspace(0, 0, 1, dtype = torch.float32) # creates 1d tensor from start to end with num_timesteps entries

    true_noise = torch.randn(512,512,3)
    true_noise = true_noise[None,:,:,:]

    # CUDA POWER
    # device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    # print(f"using {device} device")
    # u_net = u_net.to(device)
    
    # u_net_tensor = u_net(image_as_tensor.cuda())
    # print(u_net.parameters)
    # u_net_image = u_net_tensor[0].permute(1,2,0).cpu().detach().numpy()
    # print(u_net_image)
    # plt.imshow(u_net_image)

    # print(t.size())
    # diffuser = guassian_diffusion(num_timesteps=timesteps)

    # TEST = diffuser.q_sample(image_as_tensor, t)
    # plt.imshow(TEST[0].permute(1,2,0))

    # for i in range(10):  
    #     TEST = diffuser.q_sample(TEST, t)
    #     plt.imshow(TEST[0].permute(1,2,0))

    # plt.imshow(TEST[0].permute(1,2,0))
    print("done")