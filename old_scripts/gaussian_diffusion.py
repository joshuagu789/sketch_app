from torch import nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 

from lib.helpers import extract, tensor_to_negative_one_to_one, tensor_to_zero_to_one, get_index_from_list

class guassian_diffusion():
    """
    From
    https://www.assemblyai.com/blog/minimagen-build-your-own-imagen-text-to-image-model/


    """
    @torch.no_grad
    def __init__(self, *, num_timesteps: int):
        super().__init__()
        
        self.num_timesteps = num_timesteps

        # using info from assemblyai website which used DDPM paper
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype = torch.float32) # creates 1d tensor from start to end with num_timesteps entries

        alphas = 1. - self.betas
        # are these buffers?
        self.alphas_cumprod = torch.cumprod(alphas,axis=0)    # returns cumulative product of alphas as same size tensor/vector
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1-self.alphas_cumprod)

        # COPY PASTA
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value = 1.)     # pads vector with some stuff
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    @torch.no_grad
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None):
        """
        For forward diffusion and noise

        x_start is Pytorch tensor of shape (b,c,h,w)
        t is tensor of shape (b,) for timestep to noise each image to 
        noise allows option for custom noise rather than Guassian noise

        """
        actual_noise = torch.randn_like(x_start)    # returns tensor of dimensions of x_start except each entry is random number with mean 0 and var 1

        if(noise is not None):
            actual_noise = noise
        
        return extract(self.sqrt_alphas_cumprod.to("cuda"), t.to("cuda"), x_start.shape) * x_start.to("cuda") + extract(self.sqrt_one_minus_alphas_cumprod.to("cuda"), t.to("cuda"), x_start.shape) * actual_noise.to("cuda")
        
        # """
        # NOTE: below code doesnt use extract() from helpers class but does exactly what extract intends to do
        # """
        # b, *_ = t.shape     # what does *_ do?

        # -1 means use the last dimension instead which is dim = 0 for 1d tensor??
        # in this case gather takes all values of 1d column tensor sqrt of cum product of alphas at each index of t where each timestep occurs- less computing required?
        # out1 = self.sqrt_alphas_cumprod.gather(-1,t.to(torch.int64))  # EXPERIMENTAL: CONVERTED t TO INT64
        # summation_sqrt_alphas_cumprod = out1.reshape(b, *((1,) * (len(x_start.shape)-1)))   #most important line but difficulty understanding- something must have been doned iteratively

        # out2 = (1. - self.sqrt_alphas_cumprod).gather(-1,t.to(torch.int64))  # EXPERIMENTAL: CONVERTED t TO INT64
        # summation_one_minus_sqrt_alphas_cumprod = out2.reshape(b, *((1,) * (len(x_start.shape)-1)))

        # after_applying_noise = summation_sqrt_alphas_cumprod * x_start + summation_one_minus_sqrt_alphas_cumprod * actual_noise
        # # for i in range(iterations-1):
        # #     after_applying_noise = after_applying_noise + summation_sqrt_alphas_cumprod * x_start + summation_one_minus_sqrt_alphas_cumprod * actual_noise
        
        # return after_applying_noise
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        Subtracting noise (predicted by unet) from x_t to predict x_0 (initial x) where both noisy image and predicted noise have scale based off of timestep
        """
        sqrt_reciprocal_alphas_cumprod = torch.sqrt(torch.reciprocal(self.alphas_cumprod))
        sqrt_reciprocal_of_alphas_cumprod_minus_one = torch.sqrt(torch.reciprocal(self.alphas_cumprod)-1.)

                            # represents noisy image                                        represents predicted noise
        return extract(sqrt_reciprocal_alphas_cumprod,t,x_t.shape) * x_t - extract(sqrt_reciprocal_of_alphas_cumprod_minus_one, t, x_t.shape) * noise

    # def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, **kwargs):
    #     """
    #     Calculates & returns the posterior mean and variance- this represents the distribution to denoise the noisy image step by step
    #     """
    #     posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)    # see formulas online ddpm paper
    #     posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    #     posterior_mean = extract(posterior_mean_coef1, t, x_t.shape) * x_start + extract(posterior_mean_coef2, t, x_t.shape) * x_t

    #     posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    #     posterior_variance = extract(posterior_variance, t, x_t.shape)

    #                                                 # prevents posterior variance from being zero at start (or anywhere)
    #     posterior_log_variance_clipped = extract( torch.log(posterior_variance.clamp(min=1e-20)) , t, x_t)

    #     return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def denoise(self, predictor, x_t = None):
        """
        x_t should be 4d, below code assumes batch size of 1
        """
        if x_t is None:
            x_t = torch.randn(size=(1,3,224,224))
        
        image = x_t
        t = torch.linspace(0, 0, 1, dtype = torch.float32)
        t = t.to(torch.int64)
        t = t.to("cuda")
        image = image.to("cuda")

        for x in range(1000):
            prediction = predictor(tensor_to_negative_one_to_one(image).to("cuda"))    #beware convert to 0-1 in between calculations?
            image = self.predict_start_from_noise(prediction.to("cpu"), t.to("cpu"), image.to("cpu"))
            if x % 997 == 0:
                plt.imshow(tensor_to_zero_to_one(image[0]).permute(1,2,0).cpu().detach().numpy())
        return image

    # COPY PASTAD
    @torch.no_grad()
    def sample_timestep(self, model, x, t):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        x = x.to("cuda")
        betas_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t.to("cuda") * (
            # x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
            x - betas_t.to("cuda") * model(x.to("cuda"), t.to("cuda")) / sqrt_one_minus_alphas_cumprod_t.to("cuda")
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)
        
        if t == 0:
            # As pointed out by Luis Pereira (see YouTube comment)
            # The t's are offset from the t's in the paper
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t).to("cuda") * noise.to("cuda")
    #TEST
    @torch.no_grad()
    def dummy_sample_timestep(self, model, x, t):
        x = x.to("cuda")
        return x - model(x)
    
if __name__ == "__main__":

    # betas = torch.linspace(2, 20, 10, dtype = torch.float32)
    # print(betas)
    # print(5. - betas)
    # # print(betas.shape)
    # print(torch.cumprod(betas,axis=0))
    print("done")
