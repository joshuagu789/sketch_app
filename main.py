from PIL import Image
from diffusers.utils import load_image
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import opendatasets as od
import os
import glob

from gaussian_diffusion import guassian_diffusion
from u_net_no_conditioning import u_net_no_conditioning

if __name__ == "__main__":
    
    timesteps = 1000
    batch_size = 64
    learn_rate = 1e-3
    epochs = 3
    rawdataset = []
    pil_to_tensor = transforms.ToTensor()
    resizer = transforms.Resize((512,512))
    counter = 0
    # for filename in glob.glob("C:\Users\joshu\Documents\GitHub\sketch_app/*.jpg"):
    # for filename in glob.glob("./animal-image-dataset-90-different-animals/animals/*.jpg"):
    for directory_name in os.listdir("animal-image-dataset-90-different-animals/animals/animals"):
        for filename in glob.glob(os.path.join("animal-image-dataset-90-different-animals/animals/animals/" + str(directory_name),"*.jpg")):
            # if counter <= 50:
            img = Image.open(filename)
            image_as_tensor = pil_to_tensor(img)
            image_as_tensor = resizer(image_as_tensor)
            image_as_tensor = image_as_tensor.to(torch.float32)
            rawdataset.append(image_as_tensor)
            print(counter)
            counter += 1
    print(len(rawdataset))

    tensordataset = torch.stack(rawdataset)    # from array of 3d tensors into 4d tensor
    print(tensordataset.size())
    torch.save(tensordataset, "ninety-animals.pt")
    plt.imshow(tensordataset[541].permute(1,2,0).cpu().detach().numpy())
    plt.imshow(tensordataset[1331].permute(1,2,0).cpu().detach().numpy())
    plt.imshow(tensordataset[3911].permute(1,2,0).cpu().detach().numpy())
    plt.imshow(tensordataset[4912].permute(1,2,0).cpu().detach().numpy())

    od.download(
        "https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals"
    )
    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )

    image_as_tensor = pil_to_tensor(image)
    image_as_tensor = image_as_tensor[None,:,:,:]
    image_as_tensor = image_as_tensor.to(torch.float32)

    # t = torch.linspace(0, 0, 1, dtype = torch.float32) # creates 1d tensor from start to end with num_timesteps entries
    t = torch.randint(0, timesteps, (batch_size,))

    true_noise = torch.randn(512,512,3)
    # plt.imshow(true_noise)
    # print(image_as_tensor)
    true_noise = true_noise[None,:,:,:]
    u_net = u_net_no_conditioning()

    # CUDA POWER
    # device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    # print(f"using {device} device")
    # u_net = u_net.to(device)
    
    # u_net_tensor = u_net(image_as_tensor.cuda())
    # print(u_net.parameters)
    # u_net_image = u_net_tensor[0].permute(1,2,0).cpu().detach().numpy()
    # print(u_net_image)
    # plt.imshow(u_net_image)

    #dataloader = DataLoader(, batch_size = batch_size, shuffle=True)

    print(t.size())
    diffuser = guassian_diffusion(num_timesteps=timesteps)

    TEST = diffuser.q_sample(image_as_tensor, t)
    plt.imshow(TEST[0].permute(1,2,0))

    for i in range(10):  
        TEST = diffuser.q_sample(TEST, t)
        plt.imshow(TEST[0].permute(1,2,0))

    plt.imshow(TEST[0].permute(1,2,0))
    print("done")