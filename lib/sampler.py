from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt

from lib.guassian_diffusion_copy_pasta import sample_timestep, forward_diffusion_sample
from lib.helpers import tensor_to_negative_one_to_one, tensor_to_zero_to_one, show_tensor_image
from lib.image_editing import cartoonify

from io import BytesIO
import base64
import json

@torch.no_grad()
def sample_img_from_request(image_file, model, device, timesteps, width_and_height):
    """
    Same thing as sample_img but from request from website
    """
    encodings = image_file.split(b',')[-1]
    image_bytes = base64.decodebytes(encodings) 
    pil_image = Image.open(BytesIO(image_bytes))

    resizer = transforms.Resize((width_and_height,width_and_height)) 
    pil_to_tensor = transforms.ToTensor()
    image_as_tensor = pil_to_tensor(pil_image)
    image_as_tensor = resizer(image_as_tensor)
    image_as_tensor = image_as_tensor.to(torch.float32)
    image_as_tensor = image_as_tensor[None,:,:,:]

    t = torch.randint(timesteps-1, timesteps, (image_as_tensor.size(dim=0),))
    img, a = forward_diffusion_sample(tensor_to_negative_one_to_one(image_as_tensor), t, device)
    corrupted = img

    """
    for loop pretty much copied from https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=k13hj2mciCHA
    """
    for i in range(0,timesteps)[::-1]:
        print(i)
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)

    cartoon = cartoonify(tensor_to_zero_to_one(img[0]))
    plt.axis('off')
    plt.subplot(1,4,1)
    show_tensor_image(tensor_to_negative_one_to_one(image_as_tensor.detach().cpu()))
    plt.title("Original")
    plt.subplot(1,4,2)
    show_tensor_image(tensor_to_negative_one_to_one(corrupted.detach().cpu()))
    plt.title("Corrupted")
    plt.subplot(1,4,3)
    show_tensor_image(img.detach().cpu())
    plt.title("Model Output")
    plt.subplot(1,4,4)
    plt.imshow(cartoon.cpu().detach().numpy())
    plt.title("Cartoon")    
    # plt.show()
    plt.savefig("output.jpg")

@torch.no_grad()
def sample_img(path, model, device, timesteps, width_and_height):
    """
    NOTE: originally designed for testing locally without website
    path: string that is path to conditioning image jpg, expects jpg to be square with values 0-1
    model: unet
    timesteps: how noisy to make the conditioning image
    """
    resizer = transforms.Resize((width_and_height,width_and_height)) 
    pil_to_tensor = transforms.ToTensor()

    img = Image.open(path)
    image_as_tensor = pil_to_tensor(img)
    image_as_tensor = resizer(image_as_tensor)
    image_as_tensor = image_as_tensor.to(torch.float32)
    image_as_tensor = image_as_tensor[None,:,:,:]

    t = torch.randint(timesteps-1, timesteps, (image_as_tensor.size(dim=0),))
    img, a = forward_diffusion_sample(tensor_to_negative_one_to_one(image_as_tensor), t, device)
    corrupted = img
    T = timesteps

    # img = torch.randn((1, 3, img_size, img_size), device=device)
    # plt.figure(figsize=(15,15))
    # num_images = 5
    # stepsize = int(T/num_images)

    """
    for loop pretty much copied from https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=k13hj2mciCHA
    """
    for i in range(0,T)[::-1]:
        print(i)
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        # if i == 0:
        #     plt.imshow(tensor_to_zero_to_one(img[0].permute(1,2,0).cpu().detach().numpy()))      
        # if i % stepsize == 0:
        #     plt.subplot(1, num_images, int(i/stepsize)+1)
        #     show_tensor_image(img.detach().cpu())
    cartoon = cartoonify(tensor_to_zero_to_one(img[0]))
    plt.axis('off')
    plt.subplot(1,4,1)
    show_tensor_image(tensor_to_negative_one_to_one(image_as_tensor.detach().cpu()))
    plt.title("Original")
    plt.subplot(1,4,2)
    show_tensor_image(tensor_to_negative_one_to_one(corrupted.detach().cpu()))
    plt.title("Corrupted")
    plt.subplot(1,4,3)
    show_tensor_image(img.detach().cpu())
    plt.title("Model Output")
    plt.subplot(1,4,4)
    plt.imshow(cartoon.cpu().detach().numpy())
    plt.title("Cartoon")    
    plt.show()
