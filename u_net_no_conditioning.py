from torch import nn 
import torch
import torchvision

class u_net_no_conditioning(nn.Module):
    def __init__(self):
        super().__init__()
        # Contracting path
        self.down_conv1 = encoder_block(in_channels = 3, out_channels=64, dropout_factor = 0.25)
        self.down_conv2 = encoder_block(in_channels = 64, out_channels=128, dropout_factor = 0.5)
        self.down_conv3 = encoder_block(in_channels = 128, out_channels=256, dropout_factor = 0.5)
        self.down_conv4 = encoder_block(in_channels = 256, out_channels=512, dropout_factor = 0.5)
        
        # Bottleneck
        self.bottleneck1 = nn.Conv2d(512,1024, (3, 3), padding="same")
        self.relu = nn.ReLU()    # remember to do twice
        self.bottleneck2 = nn.Conv2d(1024,1024, (3, 3), padding="same")

        # Expanding path
        self.up_conv1 = decoder_block(in_channels = 1024, out_channels=512, dropout_factor = 0.5)   #skip features = s4
        self.up_conv2 = decoder_block(in_channels = 512, out_channels=256, dropout_factor = 0.5)   #skip features = s3
        self.up_conv3 = decoder_block(in_channels = 256, out_channels=128, dropout_factor = 0.5)   #skip features = s2
        self.up_conv4 = decoder_block(in_channels = 128, out_channels=64, dropout_factor = 0.5)   #skip features = s1
    
        self.final_conv = nn.Conv2d(64,3,kernel_size=1) 

    def forward(self, inputs: torch.Tensor):
        # print(inputs.shape)
        x1 = self.down_conv1(inputs)
        # print(x1.shape)
        x2 = self.down_conv2(x1)
        # print(x2.shape)
        x3 = self.down_conv3(x2)
        # print(x3.shape)
        x4 = self.down_conv4(x3)
        # print(x4.shape)

        b = self.bottleneck1(x4)
        # print(b.shape)
        b = self.relu(b)
        b = self.bottleneck2(b)
        # print(b.shape)
        b = self.relu(b)

        x = self.up_conv1(b, x4)
        # print(x.shape)
        x = self.up_conv2(x, x3)
        # print(x.shape)
        x = self.up_conv3(x, x2)
        # print(x.shape)
        x = self.up_conv4(x, x1)
        # print(x.shape)

        x = self.final_conv(x)
        # print(x.shape)

        return x
    
class encoder_block(nn.Module):
    """
    Encoder block for U Net

    :param inputs: is a tensor of (b, c, h, w) for batch, channels, height width
    :param in_channels: number of channels or depth at each pixel
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_factor):
        super().__init__()

        # performs conv and relu twice
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels = out_channels, kernel_size=3, padding=1) # replaced padding = same with padding = 1, kernel size (3,3) with 3 
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2)  # also seen input as just 2
        # self.dropout = nn.Dropout(dropout_factor)     # temp disabling this

    def forward(self, inputs: torch.Tensor):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # x = self.dropout(x)
        return x

class decoder_block(nn.Module):
    """
    Encoder block for U Net

    :param inputs: is a tensor of (b, c, h, w) for batch, channels, height width
    :param in_channels: number of channels or depth at each pixel
    :param skip_features: for concatenation part
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_factor):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels = in_channels, out_channels=out_channels, kernel_size = (2,2), stride = (2,2))  # removed padding = same from three below
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size = (3,3))
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels=out_channels, kernel_size = (3,3))
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout_factor) # temp disabling this

    def forward(self, inputs: torch.Tensor, skip_features: torch.Tensor):
        x = self.deconv(inputs)
        contracting_x = torchvision.transforms.functional.center_crop(skip_features, [x.shape[2], x.shape[3]])

        # x = torch.cat((x,skip_features))    # concatenate feature map?
        x = torch.cat([x, contracting_x], dim=1)
        # x = self.dropout(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x

    # def u_net(self, input_layer):
    #     """
    #     NOTE: taken from https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
    #             - CHANCES ARE THIS IS A UNET MODEL WITH NO CONDITIONING 
    #             - DOES THIS ASSUME INPUT IMAGE HAS WIDTH AND HEIGHT 572 AND ONE CHANNEL FOR GRAYSCALE???
    #     """

    #     print("beginning unet")
    #     print(input_layer.shape)
    #     # Contracting path
    #     s1 = self.encoder_block(inputs = input_layer, in_channels = 3, out_channels=64, dropout_factor = 0.25)  # actually 3 for in channel rgb not grayscale
    #     print(s1.shape)
    #     s2 = self.encoder_block(inputs = s1, in_channels = 64, out_channels=128, dropout_factor = 0.5)
    #     print(s2.shape)
    #     s3 = self.encoder_block(inputs = s2, in_channels = 128, out_channels=256, dropout_factor = 0.5)
    #     print(s3.shape)
    #     s4 = self.encoder_block(inputs = s3, in_channels = 256, out_channels=512, dropout_factor = 0.5)
    #     print(s4.shape)

    #     # Bottleneck
    #     conv_bottleneck1 = nn.Conv2d(512,1024, (3, 3), padding="same")   #512 or 1024?
    #     conv_bottleneck2 = nn.Conv2d(1024,1024, (3, 3), padding="same")   #512 or 1024?
    #     relu = nn.ReLU()

    #     b1 = conv_bottleneck1(s4)
    #     b1 = relu(b1)
    #     b1 = conv_bottleneck2(b1)
    #     b1 = relu(b1)
    #     print(b1.shape)

    #     # Expansive path
    #     s5 = self.decoder_block(inputs = b1, in_channels = 1024, out_channels=512, dropout_factor = 0.5, skip_features = s4)
    #     print(s5.shape)
    #     s6 = self.decoder_block(inputs = s5, in_channels = 512, out_channels=256, dropout_factor = 0.5, skip_features = s3)
    #     print(s6.shape)
    #     s7 = self.decoder_block(inputs = s6, in_channels = 256, out_channels=128, dropout_factor = 0.5, skip_features = s2)
    #     print(s7.shape)
    #     s8 = self.decoder_block(inputs = s7, in_channels = 128, out_channels=64, dropout_factor = 0.5, skip_features = s1)
    #     print(s8.shape)

    #     final_conv = nn.Conv2d(64,3,kernel_size=1) 
    #     out = final_conv(s8)
    #     print(out.shape)

    #     return out  # final shape [x, 3, 452, 452]

