import torch
import torch.nn as nn
import torch.nn.functional as F

class I_ResNet(nn.Module):
    def __init__(self, upscale_factor=4, n_filters=8):
        super(I_ResNet, self).__init__()

        # First Block
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=n_filters, kernel_size=3, padding=1)
        
        # Second Block
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        
        # Third Block
        self.conv4 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        
        # Fourth Block
        self.conv6 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        
        # Fifth Block
        self.conv8 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv9 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        
        # Final Convolution
        self.conv10 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)

        self.bilinear = nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=True)
        
        self.conv11 = nn.Conv2d(in_channels=n_filters, out_channels=2, kernel_size=(7, 7), padding=(3, 3))
        
    def forward(self, x):
        # First Block
        out1 = self.conv1(x)
        
        # Second Block
        out2 = self.conv2(out1)
        out2 = self.relu1(out2)
        out2 = self.conv3(out2)
        add1 = out1 + out2
        
        # Third Block
        out3 = self.conv4(out2)
        out3 = self.relu2(out3)
        out3 = self.conv5(out3)
        add2 = add1 + out3
        
        # Fourth Block
        out4 = self.conv6(out3)
        out4 = self.relu3(out4)
        out4 = self.conv7(out4)
        add3 = add2 + out4
        
        # Fifth Block
        out5 = self.conv8(out4)
        out5 = self.relu4(out5)
        out5 = self.conv9(out5)
        add4 = add3 + out5
        
        # Final Convolution
        out6 = self.conv10(out5)
        
        # Addition Layer
        out_final = add4 + out6
        
        # Transposed Convolution
        out_upsampled = self.bilinear(out_final)
        
        out_resized = self.conv11(out_upsampled)
        return out_resized

if __name__ == '__main__':
    input_freq = 108
    input_spatial = 32
    input_chan = 2
    upscale_factor = 4
    n_filters = 8
    
    x = torch.randn(1, input_chan, input_freq, input_spatial)
    model = I_ResNet(upscale_factor=upscale_factor, n_filters=n_filters)
    
    x_out = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {x_out.shape}')
    print(f'Upscale factor: {upscale_factor}')
