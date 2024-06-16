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
    TESTING MODEL
    """

    # od.download(
    #     "https://www.kaggle.com/datasets/andrewmvd/animal-faces"
    # )    
    # a = image_loader.store_images_from_directory_as_tensor("datasets/animal-faces/afhq/train/", loads_everything=True, image_limit=-1, width_and_height=80)
    # print(a.size())
    # plt.imshow(a[3189].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(a[8111].permute(1,2,0).cpu().detach().numpy())
    # torch.save(a, "saved_stuff/cat_and_dog_faces.pt")

    """sampling test"""
    # test_numpy = tensor_data[0].cpu().detach().numpy()
    # test_numpy2 = (np.moveaxis(test_numpy, 0, -1) * 255).astype(np.uint8)

    # cv2.imshow("a", test_numpy2)
    # test1 = cv2.Canny(test_numpy2, 50, 100)
    # plt.imshow(test1)

    # original_image = tensor_data[0]
    # original_image_4d = original_image[None,:,:,:]
    # # plt.imshow(original_image_4d[0].permute(1,2,0).cpu().detach().numpy())
    # test_prediction = u_net(tensor_to_negative_one_to_one(original_image_4d).to("cuda"))
    # test = original_image_4d.to("cuda") - test_prediction
    # plt.imshow(tensor_to_zero_to_one(test_prediction[0].permute(1,2,0).cpu().detach().numpy()))

    # t = torch.randint(999, 1000, (1,))

    # noised = noise_tool.q_sample(tensor_to_negative_one_to_one(original_image_4d),t=t)
    # denoised = torch.randn((1,3,224,224))
    # plt.figure(figsize=(15,15))
    # plt.axis('off')

    # denoised = noise_tool.sample_timestep(model=u_net,x=noised,t=t)   # evaluate model
    # plt.imshow(tensor_to_zero_to_one(denoised[0].permute(1,2,0).cpu().detach().numpy()))

    sample_img("test_images/test_img_02.jpg", u_net, device, 200, 80)
    sample_img("test_images/test_img_01.jpg", u_net, device, 250, 80)
    sample_img("datasets/animal-faces/afhq/train/cat/flickr_cat_000002.jpg", u_net, device, 250, 80)
    sample_img("datasets/animal-faces/afhq/val/cat/flickr_cat_000351.jpg", u_net, device, 250, 80)

    chosen_img = tensor_data[0]
    chosen_img = chosen_img[None,:,:,:]
    t = torch.randint(249, 250, (chosen_img.size(dim=0),))
    # plt.imshow((chosen_img[0]).permute(1,2,0).cpu().detach().numpy())
    img, a = forward_diffusion_sample(tensor_to_negative_one_to_one(chosen_img), t, device)
    corrupted = img
    T = 250
    print(img.shape)
    # plt.imshow(tensor_to_zero_to_one(img[0]).permute(1,2,0).cpu().detach().numpy())

    img_size = 80
    # img = torch.randn((1, 3, img_size, img_size), device=device)
    # plt.figure(figsize=(15,15))
    # num_images = 5
    # stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        print(i)
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(u_net, img, t)
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
    show_tensor_image(tensor_to_negative_one_to_one(chosen_img.detach().cpu()))
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
    
    """
    Loading raw jpg images and saving them as 4d tensor
    """
    # od.download(
    #     "https://www.kaggle.com/datasets/gpiosenka/100-bird-species"
    # )    
    # od.download(
    #     "https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals"
    # )

    # tensordataset = image_loader.store_images_from_directory_as_tensor("100-bird-species/train/HARPY EAGLE", image_limit=10)
    # print(tensordataset.size())
    # torch.save(tensordataset, "harpy_eagles01.pt")
    # plt.imshow(tensordataset[5].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[53].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[97].permute(1,2,0).cpu().detach().numpy())
    # plt.imshow(tensordataset[167].permute(1,2,0).cpu().detach().numpy())
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