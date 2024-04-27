from torch import nn
import torch
import torch.nn.functional as F

from helpers import extract

class guassian_diffusion(nn.Module):
    """
    From
    https://www.assemblyai.com/blog/minimagen-build-your-own-imagen-text-to-image-model/


    """
    def __init__(self, *, num_timesteps: int):
        super().__init__()
        
        self.num_timesteps = num_timesteps

        # using info from assemblyai website which used DDPM paper
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype = torch.float32) # creates 1d tensor from start to end with num_timesteps entries

        alphas = 1. - betas
        # are these buffers?
        self.alphas_cumprod = torch.cumprod(alphas,axis=0)    # returns cumulative product of alphas as same size tensor/vector
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value = 1.)     # pads vector with some stuff

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None, iterations = 1):
        """
        For forward diffusion and noise

        x_start is Pytorch tensor of shape (b,c,h,w)
        t is tensor of shape (b,) for timestep to noise each image to 
        noise allows option for custom noise rather than Guassian noise

        """
        actual_noise = torch.randn_like(x_start)    # returns tensor of dimensions of x_start except each entry is random number with mean 0 and var 1

        if(noise is not None):
            actual_noise = noise
        
        """
        NOTE: below code doesnt use extract() from helpers class but does exactly what extract intends to do
        """
        b, *_ = t.shape     # what does *_ do?

        # -1 means use the last dimension instead which is dim = 0 for 1d tensor??
        # in this case gather takes all values of 1d column tensor sqrt of cum product of alphas at each index of t where each timestep occurs- less computing required?
        out1 = self.sqrt_alphas_cumprod.gather(-1,t.to(torch.int64))  # EXPERIMENTAL: CONVERTED t TO INT64
        summation_sqrt_alphas_cumprod = out1.reshape(b, *((1,) * (len(x_start.shape)-1)))   #most important line but difficulty understanding- something must have been doned iteratively

        out2 = (1. - self.sqrt_alphas_cumprod).gather(-1,t.to(torch.int64))  # EXPERIMENTAL: CONVERTED t TO INT64
        summation_one_minus_sqrt_alphas_cumprod = out2.reshape(b, *((1,) * (len(x_start.shape)-1)))

        after_applying_noise = summation_sqrt_alphas_cumprod * x_start + summation_one_minus_sqrt_alphas_cumprod * actual_noise
        # for i in range(iterations-1):
        #     after_applying_noise = after_applying_noise + summation_sqrt_alphas_cumprod * x_start + summation_one_minus_sqrt_alphas_cumprod * actual_noise
        
        return after_applying_noise
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        Subtracting noise (predicted by unet) from x_t to predict x_0 (initial x) where both noisy image and predicted noise have scale based off of timestep
        """
        sqrt_reciprocal_alphas_cumprod = torch.sqrt(torch.reciprocal(self.alphas_cumprod))
        sqrt_reciprocal_of_alphas_cumprod_minus_one = torch.sqrt(torch.reciprocal(self.alphas_cumprod)-1.)

                            # represents noisy image                                        represents predicted noise
        return extract(sqrt_reciprocal_alphas_cumprod,t,x_t.shape) * x_t - extract(sqrt_reciprocal_of_alphas_cumprod_minus_one, t, x_t.shape) * noise

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Calculates & returns the posterior mean and variance- this represents the distribution to denoise the noisy image step by step
        """
        posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)    # see formulas online ddpm paper
        posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        posterior_mean = extract(posterior_mean_coef1, t, x_t.shape) * x_start + extract(posterior_mean_coef2, t, x_t.shape) * x_t

        posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        posterior_variance = extract(posterior_variance, t, x_t.shape)

                                                    # prevents posterior variance from being zero at start (or anywhere)
        posterior_log_variance_clipped = extract( torch.log(posterior_variance.clamp(min=1e-20)) , t, x_t)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

if __name__ == "__main__":

    betas = torch.linspace(2, 20, 10, dtype = torch.float32)
    print(betas)
    print(5. - betas)
    # print(betas.shape)
    print(torch.cumprod(betas,axis=0))
    print("done")
