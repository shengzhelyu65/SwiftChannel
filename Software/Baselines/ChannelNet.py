import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, channels, kernel_size=5, padding=2)
        
        init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='linear')

        if self.conv1.bias is not None:
            init.constant_(self.conv1.bias, 0)
        if self.conv2.bias is not None:
            init.constant_(self.conv2.bias, 0)
        if self.conv3.bias is not None:
            init.constant_(self.conv3.bias, 0)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=20):
        super(DnCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64, eps=1e-3),
                    nn.ReLU(inplace=True)
                ) for _ in range(num_of_layers - 2)
            ]
        )
        
        self.conv_last = nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.hidden_layers(out)
        out = self.conv_last(out)
        out = x - out
        return out
    
class ChannelNet(nn.Module):
    def __init__(self, channels=2, scale_factor=4):
        super(ChannelNet, self).__init__()
        self.srcnn = SRCNN(channels=channels)
        self.dncnn = DnCNN(channels=channels)
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.srcnn(x)
        x = self.dncnn(x)
        
        return x

if __name__ == '__main__':
    input_freq = 108
    input_spatial = 32
    input_chan = 2
    
    x = torch.randn(1, input_chan, input_freq, input_spatial)
    model = ChannelNet(channels=input_chan, scale_factor=4)
    
    x_out = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {x_out.shape}')