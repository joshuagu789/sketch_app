from PIL import Image
from diffusers.utils import load_image
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from gaussian_diffusion import guassian_diffusion
from u_net_no_conditioning import u_net_no_conditioning

if __name__ == "__main__":
    
    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )

    pil_to_tensor = transforms.ToTensor()
    image_as_tensor = pil_to_tensor(image)
    image_as_tensor = image_as_tensor[None,:,:,:]
    image_as_tensor = image_as_tensor.to(torch.float32)

    t = torch.linspace(0, 0, 1, dtype = torch.float32) # creates 1d tensor from start to end with num_timesteps entries

    true_noise = torch.randn(512,512,3)
    # plt.imshow(true_noise)
    
    true_noise = true_noise[None,:,:,:]
    u_net = u_net_no_conditioning()
    u_net_image = u_net.u_net(input_layer=image_as_tensor)
    print(u_net_image.shape)
    image_uses_grad = u_net_image[0].permute(1,2,0)
    plt.imshow(image_uses_grad)
    plt.imshow(u_net_image[0].permute(1,2,0).detach().numpy())

    # diffuser = guassian_diffusion(num_timesteps=1)
    # TEST = diffuser.q_sample(image_as_tensor, t)

    # for i in range(20):
    #     TEST = diffuser.q_sample(TEST, t)

    # plt.imshow(TEST[0].permute(1,2,0))
    print("done")