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
    image_as_tensor = image_as_tensor.to(torch.float64)

    t = torch.linspace(0, 0, 1, dtype = torch.float64) # creates 1d tensor from start to end with num_timesteps entries
    # print(t.shape)

    # print(image_as_tensor)
    # print(image_as_tensor.shape)
    # plt.imshow(image_as_tensor.permute(1,2,0))
    # print(image_as_tensor)

    true_noise = torch.randn(512,512,3)
    true_noise = true_noise[None,:,:,:]
    u_net = u_net_no_conditioning()
    u_net.u_net(input_layer=true_noise)

    # print(true_noise)
    # plt.imshow(true_noise)

    diffuser = guassian_diffusion(num_timesteps=1)
    TEST = diffuser.q_sample(image_as_tensor, t)

    for i in range(100):
        TEST = diffuser.q_sample(TEST, t)

    print(TEST)
    print(TEST.shape)

    plt.imshow(TEST[0].permute(1,2,0))
    print("done")