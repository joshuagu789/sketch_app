from torchvision import transforms
import os
import glob
from PIL import Image
import torch

class image_loader():
    def __init__(self):
        self.pil_to_tensor = transforms.ToTensor()

    def load_img_server(self, img):
        """
        Takes in jpg from server and returns a 4D tensor of shape (batch, channels, width, height)
        """
        return

    def store_images_from_directory_as_tensor(self, path, loads_everything = False, image_limit = -1, width_and_height = 224):
        """
        Returns a 4d tensor of (number, channels, width, height) from path

        path:
        loads_everything: if going to load every image in subdirectories of a directory
        image_limit: -1 if no limit
        width_and_height: 
        """
        rawdataset = []
        resizer = transforms.Resize((width_and_height,width_and_height)) 
        counter = 0

        if loads_everything:
            for directory_name in os.listdir(path):
                for filename in glob.glob(os.path.join(path + str(directory_name),"*.jpg")):
                    # if counter <= 50:
                    img = Image.open(filename)
                    image_as_tensor = self.pil_to_tensor(img)
                    image_as_tensor = resizer(image_as_tensor)
                    image_as_tensor = image_as_tensor.to(torch.float32)
                    rawdataset.append(image_as_tensor)
                    if(image_limit > 0):
                        counter += 1
                        if(counter == image_limit):
                            break          
        else:
            for filename in glob.glob(os.path.join(path,"*.jpg")):
                img = Image.open(filename)
                image_as_tensor = self.pil_to_tensor(img)
                image_as_tensor = resizer(image_as_tensor)
                image_as_tensor = image_as_tensor.to(torch.float32)
                rawdataset.append(image_as_tensor)
                if(image_limit > 0):
                    counter += 1
                    if(counter == image_limit):
                        break   
        return torch.stack(rawdataset)