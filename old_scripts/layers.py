from torch import nn 
import torch 
from einops import rearrange 
import math
# from einops_exts.torch import EinopsToAndFrom

class SinusoidalPosEmb(nn.Module):
    '''
    COPY PASTED WILL REVIEW WHAT THIS DOES LATER

    Generates sinusoidal positional embedding tensor. In this case, position corresponds to time. For more information
        on sinusoidal embeddings, see ["Positional Encoding - Additional Details"](https://www.assemblyai.com/blog/how-imagen-actually-works/#timestep-conditioning).
    '''

    def __init__(self, dim: int):
        """
        :param dim: Dimensionality of the embedding space
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: Tensor of positions (i.e. times) to generate embeddings for.
        :return: T x D tensor where T is the number of positions/times and D is the dimensionality of the embedding
            space
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
    
class CrossEmbedLayer(nn.Module):
    '''
    COPY PASTED WILL REVIEW WHAT THIS DOES LATER


    Module that performs cross embedding on an input image (essentially an Inception module) which maintains channel
        depth.

    E.g. If input a 64x64 image with 128 channels and use kernel_sizes = (3, 7, 15) and stride=1, then 3 convolutions
        will be performed:

        1: 64 filters, (3x3) kernel, stride=(1x1), padding=(1x1) -> 64x64 output
        2: 32 filters, (7x7) kernel, stride=(1x1), padding=(3x3) -> 64x64 output
        3: 32 filters, (15x15) kernel, stride=(1x1), padding=(7x7) -> 64x64 output

        Concatenate them for a resulting 64x64 image with 128 output channels
    '''

    def __init__(
            self,
            dim_in: int,
            kernel_sizes: tuple[int, ...],
            dim_out: int = None,
            stride: int = 2
    ):
        """
        :param dim_in: Number of channels in the input image.
        :param kernel_sizes: Tuple of kernel sizes to use for convolutions.
        :param dim_out: Number of channels in output image. Defaults to `dim_in`.
        :param stride: Stride of convolutions.
        """
        super().__init__()
        # Ensures stride and all kernels are either all odd or all even
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])

        # Set output dimensionality to be same as input if not provided
        # dim_out = default(dim_out, dim_in)
        dim_out = dim_in
        if(dim_out is not None):
            dim_out = dim_out

        # Sort the kernels by size and determine number of kernels
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # Determine number of filters for each kernel. They will sum to dim_out and be descending with kernel size
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        # Create the convolution objects
        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Perform each convolution and then concatenate the results along the channel dim.
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)
    
class ResnetBlock(nn.Module):
    """
    Mostly copied and pasted but edited some parts so that there are no errors (didnt use their helper methods and other features)

    ResNet Block.
    """
    def __init__(
            self,
            dim: int,
            dim_out: int,
            *,
            cond_dim: int = None,
            time_cond_dim: int = None,
            groups: int = 8,
    ):
        """
        :param dim: Number of channels in the input.
        :param dim_out: Number of channels in the output.
        :param cond_dim: Dimension of the conditioning tokens on which to perform cross attention with the input.
        :param time_cond_dim: Dimension of the time conditioning tensor.
        :param groups: Number of groups to use in the GroupNorms. See :class:`.Block`.
        """
        super().__init__()

        self.time_mlp = None

        # if exists(time_cond_dim):
        if time_cond_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None
        # if exists(cond_dim):
        if cond_dim is not None:
            self.cross_attn = EinopsToAndFrom(
                'b c h w',
                'b (h w) c',
                CrossAttention(
                    dim=dim_out,
                    context_dim=cond_dim
                )
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()

    def forward(self, x: torch.tensor, time_emb: torch.tensor = None, cond: torch.tensor = None) -> torch.tensor:
        """
        :param x: Input image. Shape (b, c, s, s).
        :param time_emb: Time conditioning tensor. Shape (b, c2).
        :param cond: Main conditioning tensor. Shape (b, c3).
        :return: Output image. Shape (b, c, s, s)
        """

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)

        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context=cond) + h

        h = self.block2(h, scale_shift=scale_shift)

        return h + self.res_conv(x)
