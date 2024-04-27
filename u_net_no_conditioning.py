from torch import nn 
import torch
import torchvision

# TESTTESTTEST
class u_net_no_conditioning(nn.Module):
    def __init__(self):
        self.contracting_path = nn.Sequential{
            
        }

    def encoder_block(self, inputs: torch.Tensor, in_channels: int, out_channels: int, dropout_factor):
        """
        Encoder block for U Net

        :param inputs: is a tensor of (b, c, h, w) for batch, channels, height width
        :param in_channels: number of channels or depth at each pixel
        """

        # performs conv and relu twice
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels = out_channels, kernel_size=3, padding=1) # replaced padding = same with padding = 1, kernel size (3,3) with 3 
        relu = nn.ReLU()
        conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=3, padding=1) 
        pool = nn.MaxPool2d(2)  # also seen input as just 2
        dropout = nn.Dropout(dropout_factor)

        x = conv1(inputs)
        x = relu(x)
        x = conv2(x)
        x = relu(x)
        x = pool(x)
        x = dropout(x)

        return x

    def decoder_block(self, inputs: torch.Tensor, in_channels: int, out_channels: int, dropout_factor, skip_features: torch.Tensor):
        """
        Encoder block for U Net

        :param inputs: is a tensor of (b, c, h, w) for batch, channels, height width
        :param in_channels: number of channels or depth at each pixel
        :param skip_features: for concatenation part
        """

        deconv = nn.ConvTranspose2d(in_channels = in_channels, out_channels=out_channels, kernel_size = (2,2), stride = (2,2))  # removed padding = same from three below
        conv1 = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size = (3,3))
        conv2 = nn.Conv2d(in_channels = out_channels, out_channels=out_channels, kernel_size = (3,3))
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout_factor)

        x = deconv(inputs)
        contracting_x = torchvision.transforms.functional.center_crop(skip_features, [x.shape[2], x.shape[3]])

        # x = torch.cat((x,skip_features))    # concatenate feature map?
        x = torch.cat([x, contracting_x], dim=1)
        x = dropout(x)
        x = conv1(x)
        x = relu(x)
        x = conv2(x)
        x = relu(x)

        return x

    def u_net(self, input_layer):
        """
        NOTE: taken from https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
                - CHANCES ARE THIS IS A UNET MODEL WITH NO CONDITIONING 
                - DOES THIS ASSUME INPUT IMAGE HAS WIDTH AND HEIGHT 572 AND ONE CHANNEL FOR GRAYSCALE???
        """

        print("beginning unet")
        print(input_layer.shape)
        # Contracting path
        s1 = self.encoder_block(inputs = input_layer, in_channels = 3, out_channels=64, dropout_factor = 0.25)  # actually 3 for in channel rgb not grayscale
        print(s1.shape)
        s2 = self.encoder_block(inputs = s1, in_channels = 64, out_channels=128, dropout_factor = 0.5)
        print(s2.shape)
        s3 = self.encoder_block(inputs = s2, in_channels = 128, out_channels=256, dropout_factor = 0.5)
        print(s3.shape)
        s4 = self.encoder_block(inputs = s3, in_channels = 256, out_channels=512, dropout_factor = 0.5)
        print(s4.shape)

        # Bottleneck
        conv_bottleneck1 = nn.Conv2d(512,1024, (3, 3), padding="same")   #512 or 1024?
        conv_bottleneck2 = nn.Conv2d(1024,1024, (3, 3), padding="same")   #512 or 1024?
        relu = nn.ReLU()

        b1 = conv_bottleneck1(s4)
        b1 = relu(b1)
        b1 = conv_bottleneck2(b1)
        b1 = relu(b1)
        print(b1.shape)

        # Expansive path
        s5 = self.decoder_block(inputs = b1, in_channels = 1024, out_channels=512, dropout_factor = 0.5, skip_features = s4)
        print(s5.shape)
        s6 = self.decoder_block(inputs = s5, in_channels = 512, out_channels=256, dropout_factor = 0.5, skip_features = s3)
        print(s6.shape)
        s7 = self.decoder_block(inputs = s6, in_channels = 256, out_channels=128, dropout_factor = 0.5, skip_features = s2)
        print(s7.shape)
        s8 = self.decoder_block(inputs = s7, in_channels = 128, out_channels=64, dropout_factor = 0.5, skip_features = s1)
        print(s8.shape)

        final_conv = nn.Conv2d(64,3,kernel_size=1) 
        out = final_conv(s8)
        print(out.shape)

        return out
        # unet_no_conditioning_contracting_path = nn.Sequential(
        #     nn.Conv2d(start_neurons * 1, (3, 3), padding="same"), #(input_layer),
        #     nn.ReLU(),
        #     nn.Conv2d(start_neurons * 1, (3, 3), padding="same"),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2,2)),
        #     nn.Dropout(0.25),

        #     nn.Conv2d(start_neurons * 2, (3, 3), padding="same"), #(input_layer),
        #     nn.ReLU(),
        #     nn.Conv2d(start_neurons * 2, (3, 3), padding="same"),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2,2)),
        #     nn.Dropout(0.5),

        #     nn.Conv2d(start_neurons * 4, (3, 3), padding="same"), #(input_layer),
        #     nn.ReLU(),
        #     nn.Conv2d(start_neurons * 4, (3, 3), padding="same"),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2,2)),
        #     nn.Dropout(0.5),

        #     nn.Conv2d(start_neurons * 8, (3, 3), padding="same"), #(input_layer),
        #     nn.ReLU(),
        #     nn.Conv2d(start_neurons * 8, (3, 3), padding="same"),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2,2)),
        #     nn.Dropout(0.5),
        # )

        # unet_no_conditioning_bottle_neck = nn.Sequential(
        #     nn.Conv2d(start_neurons * 16, (3, 3), padding="same"), 
        #     nn.ReLU(),
        #     nn.Conv2d(start_neurons * 16, (3, 3), padding="same"),
        #     nn.ReLU(),
        # )

        # unet_no_conditioning_expansive_path = nn.Sequential(
        #     nn.ConvTranspose2d(start_neurons * 8, (3,3), strides=(2,2),padding="same"), # Conv2dtranspose is "gradient of conv2d with respect to input"
        #     x = torch.cat([])
        # )
