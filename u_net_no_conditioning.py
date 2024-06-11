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
        # print("start: " + str(inputs.shape))
        x1 = self.down_conv1(inputs)
        # print(x1.shape)
        x2 = self.down_conv2(x1)
        # print(x2.shape)
        x3 = self.down_conv3(x2)
        # print(x3.shape)
        x4 = self.down_conv4(x3)
        # print("after downconv: " + str(x4.shape))

        b = self.bottleneck1(x4)
        # print(b.shape)
        # b = self.relu(b)
        b = self.bottleneck2(b)
        # print("bottleneck: " + str(b.shape))
        # b = self.relu(b)

        x = self.up_conv1(b, x4)
        # print(x.shape)
        x = self.up_conv2(x, x3)
        # print(x.shape)
        x = self.up_conv3(x, x2)
        # print(x.shape)
        x = self.up_conv4(x, x1)
        # print("after upconv: " + str(x.shape))

        x = self.final_conv(x)
        # print("final: " + str(x.shape))

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
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels = out_channels, kernel_size=3, padding="same") 
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=3, padding="same") 
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
        self.deconv = nn.ConvTranspose2d(in_channels = in_channels, out_channels=out_channels, kernel_size = (2,2), stride = (2,2))  
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size = (3,3), padding="same")
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels=out_channels, kernel_size = (3,3), padding="same")
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
    
class dummy_u_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding="same") 
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same") 
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same") 
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding="same") 
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same") 
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding="same") 
        self.relu = nn.ReLU()
    def forward(self, inputs: torch.Tensor):
        x = self.conv1(inputs)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        x = self.conv6(x)

        return x

class small_u_net_no_conditioning(nn.Module):
    def __init__(self):
        super().__init__()
        # Contracting path
        self.down_conv1 = encoder_block(in_channels = 3, out_channels=64, dropout_factor = 0.25)
        
        # Bottleneck
        self.bottleneck1 = nn.Conv2d(64,128, (3, 3), padding="same")
        self.relu = nn.ReLU()    # remember to do twice
        self.bottleneck2 = nn.Conv2d(128,128, (3, 3), padding="same")

        # Expanding path
        self.up_conv1 = decoder_block(in_channels = 128, out_channels=64, dropout_factor = 0.5)   #skip features = s4
    
        self.final_conv = nn.Conv2d(64,3,kernel_size=1) 

    def forward(self, inputs: torch.Tensor):
        # print("start: " + str(inputs.shape))
        x1 = self.down_conv1(inputs)
        # print("after downconv: " + str(x4.shape))

        b = self.bottleneck1(x1)
        b = self.relu(b)
        b = self.bottleneck2(b)
        # print("bottleneck: " + str(b.shape))
        b = self.relu(b)

        x = self.up_conv1(b, x1)
        # print("after upconv: " + str(x.shape))

        x = self.final_conv(x)
        # print("final: " + str(x.shape))

        return x