from torch import nn 
import torch
from layers import SinusoidalPosEmb, CrossEmbedLayer, ResnetBlock
from einops.layers.torch import Rearrange

class u_net(nn.Module):
    def __init__(self, lowres_images = False, *args, **kwargs):
        """
        :param lowres_images: whether the model will be conditioned w/ low resolution images
        """
        super().__init__()
        
        """
        Below variables are usually passed in constructor with default values but lets just keep them here w/ default values :^     )
        """
        self.dim = 128      # max number of channels that unet can have?
        self.channels = 3
        self.cond_dim = self.dim        # typically pass parameter cond_dim in init
        self.time_cond_dim = self.dim * 4 * (2 if lowres_images else 1)     # for more power???

        # constants
        NUM_TIME_TOKENS = 2

        # uses time vector's timesteps for hidden states- what are hidden states? is this for text conditioning??
        # maps time conditioning to time hidden state????
        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim=self.dim),     # not sure what this is- time step conditioning and positional encoding something something for text
            nn.Linear(self.dim, self.time_cond_dim),
            nn.SiLU()       # sigmoid linear unit
        )

        # "Maps time hidden state to time conditioning (non-attention)"
        self.to_time_cond = nn.Sequential(
            nn.Linear(self.time_cond_dim, self.time_cond_dim)
        )

        # "Maps time hidden states to time tokens for main conditioning tokens (attention)"
        self.to_time_tokens = nn.Sequential(
            nn.Linear(self.time_cond_dim, self.cond_dim * NUM_TIME_TOKENS),
            Rearrange('b (r d) -> b r d', r=NUM_TIME_TOKENS)    # increases tensor by one dimension?
        )

        """
        Modules created for unet architecture
        """
        self.initial_convolution = CrossEmbedLayer(dim_in=self.channels, dim_out = self.dim, kernel_sizes = (3,7,15), stride = 1)

        # self.initial_resnet_block = Resn
    
    def forward(self, x: torch.Tensor, time: torch.Tensor, *args, **kwargs):
        time_hiddens = self.to_time_hiddens(time)

        t = self.to_time_cond(time_hiddens)
        time_tokens = self.to_time_tokens(time_hiddens)

        x = self.initial_convolution(x)

    