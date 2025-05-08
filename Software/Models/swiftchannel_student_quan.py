import torch
from torch import nn as nn
from torch.nn import functional as F

class PixelShuffleCustom(nn.Module):
    def __init__(self, upscale_factor_height, upscale_factor_width):
        super(PixelShuffleCustom, self).__init__()
        self.upscale_factor_height = upscale_factor_height
        self.upscale_factor_width = upscale_factor_width

    def forward(self, x):
        """
        Custom pixel shuffle implementation for different scaling in frequency and antenna domains.
        Upscales `x` with separate factors for each dimension.
        """
        batch_size, channels, height, width = x.size()
        out_channels = channels // (self.upscale_factor_height * self.upscale_factor_width)
        x = x.view(batch_size, out_channels, self.upscale_factor_height, self.upscale_factor_width, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(batch_size, out_channels, self.upscale_factor_height * height, self.upscale_factor_width * width)
        return x
    
class CustomAct(nn.Module):
    def __init__(self):
        super(CustomAct, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.float_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        out_sigmoid = self.sigmoid(x)
        out = self.float_func.add_scalar(out_sigmoid, -0.5)
        return out

class SPAB(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super(SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.c2_r = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = torch.nn.ReLU(inplace=False)
        self.act2 = CustomAct()
        self.float_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        sim_att = self.act2(out2)
        
        sum = self.float_func.add(x, out2)
        
        out = self.float_func.mul(sum, sim_att)

        return out, sum, sim_att

class SwiftChannelQuan(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, middle_channels, feature_channels, downsample_channels=4, upscale=4):
        super(SwiftChannelQuan, self).__init__()
        
        if isinstance(upscale, tuple):
            upscale_height, upscale_width = upscale
        else:
            upscale_height = upscale
            upscale_width = upscale

        in_channels = num_in_ch
        out_channels = num_out_ch
        
        self.quan = torch.ao.quantization.QuantStub()
        self.dequan = torch.ao.quantization.DeQuantStub()

        self.conv_1 = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        
        self.block_1 = SPAB(in_channels=feature_channels, mid_channels=middle_channels, out_channels=feature_channels)
        self.block_2 = SPAB(in_channels=feature_channels, mid_channels=middle_channels, out_channels=feature_channels)
        self.block_3 = SPAB(in_channels=feature_channels, mid_channels=middle_channels, out_channels=feature_channels)
        self.block_4 = SPAB(in_channels=feature_channels, mid_channels=middle_channels, out_channels=feature_channels)
        
        self.conv_2 = nn.Conv2d(feature_channels, downsample_channels, kernel_size=3, padding=1)
        
        self.conv_upsample = nn.Conv2d(downsample_channels, out_channels * upscale_height * upscale_width, kernel_size=1, padding=0, stride=1)
        self.pixel_shuffle = PixelShuffleCustom(upscale_height, upscale_width)

    def forward(self, x):
        x_quan = self.quan(x)
        out_feature = self.conv_1(x_quan)
        out_b1, out_b1_sum, att1 = self.block_1(out_feature)
        out_b2, out_b2_sum, att2 = self.block_2(out_b1)
        out_b3, out_b3_sum, att3 = self.block_3(out_b2)
        out_b4, out_b4_sum, att4 = self.block_4(out_b3)
        
        out_feature2 = self.conv_2(out_b4)
        
        out_upsample_pre = self.conv_upsample(out_feature2)
        output = self.pixel_shuffle(out_upsample_pre)
        
        output_dequan = self.dequan(output)
        
        return output_dequan