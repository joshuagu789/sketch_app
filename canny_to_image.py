import torch
import numpy as np
import cv2
from PIL import Image

from transformers import pipeline
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image, make_image_grid

# import os

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
# print(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'])
# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
# else:
#     raise ValueError()

image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

image_as_tensor = np.array(object = image)   # 512 x 512 x 3

# cv2.imshow("512 x 512 x 3 with rgb", image_as_tensor)       # appears that image from tensor is more blue than image from pillow img
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# image.show()            

low_threshold = 100
high_threshold = 200

"""
Converting image from pillow image to tensor with bunch of small formating to adjust tensor
    - appears that none of the formatting affects appearance of image generated from tensor and thus values of tensor
"""

image_as_tensor = cv2.Canny(image = image_as_tensor, threshold1 = low_threshold, threshold2 = high_threshold)     # 512 x 512, converts from rgb to greyscale

# cv2.imshow("512 x 512", image_as_tensor)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_as_tensor = image_as_tensor[:,:,None]       # 512 x 512 x 1 (turned every value in 512 x 512 matrix into a 1 x 1 array containing value)

# cv2.imshow("512 x 512 x 1", image_as_tensor)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_as_tensor = np.concatenate([image_as_tensor, image_as_tensor, image_as_tensor], axis=2)   # 512 x 512 x 3?? did it take each 1 x 1 array entry and turn it into 3 x 1 array with same entry 3 times?

# cv2.imshow("512 x 512 x 3 after concatenate operation", image_as_tensor)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

canny_image = Image.fromarray(image_as_tensor)     # 512 x 512

"""
Controlnet stuff
"""
# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",torch_dtype = torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet = controlnet, torch_dtype = torch.float16)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",torch_dtype = torch.float32)

pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet = controlnet, torch_dtype = torch.float32)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)     # model offloading to reduce memory

# pipe.enable_model_cpu_offload() # all memory to cpu using accelerate
# pipe.enable_sequential_cpu_offload() # all memory to cpu using accelerate

generator = torch.manual_seed(0)    # torch.Generator object for RNG

# device = torch.device("mps")
# controlnet = controlnet.to(device)
# pipe = pipe.to(device)

output = pipe("anime woman", num_inference_steps = 20, generator = generator, image = canny_image)
output = output.images
output = output[0]

make_image_grid([image, canny_image, output], rows=1, cols=3).show("girl with a pearl earring as a farmer")

print("done")