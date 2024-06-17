from diffusers.utils import load_image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim import Adam              
from torch.utils.data import DataLoader, TensorDataset

from gaussian_diffusion import guassian_diffusion
from guassian_diffusion_copy_pasta import sample_timestep, forward_diffusion_sample
from u_net_no_conditioning import u_net_no_conditioning, dummy_u_net, small_u_net_no_conditioning
from u_net_copy_pasta import SimpleUnet
from trainer import trainer
from helpers import tensor_to_negative_one_to_one, tensor_to_zero_to_one
from image_loader import image_loader
from image_editing import cartoonify
from sampler import sample_img

if __name__ == "__main__":

    device = "cuda"
    timesteps = 500
    T = 500
    batch_size = 80
    learn_rate = 0.0005
    epochs = 40
    counter = 0
    rawdataset = []   # for process of converting jpg images into 4d tensor

    """
    Loading Model and Dataset
    """
    image_loader = image_loader()
    trainer = trainer()
    pil_to_tensor = transforms.ToTensor()
    noise_tool = guassian_diffusion(num_timesteps=timesteps)

    # u_net = dummy_u_net()
    u_net = SimpleUnet()
    # u_net = small_u_net_no_conditioning()
    u_net = u_net.to(device)
    optimizer = Adam(u_net.parameters(), lr=learn_rate)  
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=0.999)
    epoch_history = []

    state = torch.load("saved_stuff/super_copied_cat_and_dog_faces.pth")     # contains state_dict and optimizer and epoch history
    u_net.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"]) 
    for group in optimizer.param_groups:            # updating learn rate dynamically
        group["lr"] = learn_rate
    epoch_history = state["epoch_history"]

    tensor_data = torch.load("saved_stuff/cat_and_dog_faces.pt") 
    tensor_dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(tensor_dataset, batch_size = batch_size, shuffle = True, pin_memory=True)

    """
    Getting Dataset
    """

    # od.download(
    #     "https://www.kaggle.com/datasets/andrewmvd/animal-faces"
    # )    
    # a = image_loader.store_images_from_directory_as_tensor("datasets/animal-faces/afhq/train/", loads_everything=True, image_limit=-1, width_and_height=80)
    # print(a.size())
    # plt.imshow(a[3189].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(a[8111].permute(1,2,0).cpu().detach().numpy())
    # torch.save(a, "saved_stuff/cat_and_dog_faces.pt")

    """
    Evaluating Model
    """
    sample_img("test_images/test_img_03.jpg", u_net, device, 50, 80)
    sample_img("test_images/test_img_03.jpg", u_net, device, 100, 80)
    sample_img("datasets/animal-faces/afhq/train/cat/flickr_cat_000002.jpg", u_net, device, 250, 80)
    sample_img("datasets/animal-faces/afhq/val/cat/flickr_cat_000351.jpg", u_net, device, 250, 80)

    """
    Training and Saving Progress on Model
    """
    trainer.train_loop(
        noise_predictor=u_net, 
        dataloader=dataloader, 
        batch_size=batch_size, 
        timesteps=timesteps, 
        epochs=epochs, 
        device=device, 
        optimizer=optimizer, 
        scheduler=scheduler,
        epoch_history=epoch_history
    )

    state = {
        "state_dict": u_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch_history": epoch_history,
        # "scheduler": scheduler.state_dict(),
    }
    # torch.save(state, "saved_stuff/dummy_u_net_10_harpy_eagles.pth")
    torch.save(state, "saved_stuff/super_copied_cat_and_dog_faces.pth")
    
   