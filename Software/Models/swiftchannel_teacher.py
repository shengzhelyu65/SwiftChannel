import torch
from torch import nn as nn
import torch.nn.functional as F

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

def pixelshuffle_block(in_channels, out_channels, upscale_factor_height=2, upscale_factor_width=2):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(in_channels, out_channels*upscale_factor_height*upscale_factor_width, kernel_size=3, padding=1, bias=True)
    return nn.Sequential(conv, PixelShuffleCustom(upscale_factor_height, upscale_factor_width))

class PAB(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super(PAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.c2_r = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.c3_r = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.act1 = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)

        out3 = (self.c3_r(out2_act))

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att

class SwiftChannelTeacher(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, feature_channels, upscale, feature_output=False):
        super(SwiftChannelTeacher, self).__init__()
        
        if isinstance(upscale, tuple):
            upscale_height, upscale_width = upscale
        else:
            upscale_height = upscale
            upscale_width = upscale

        in_channels = num_in_ch
        out_channels = num_out_ch

        self.conv_1 = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.block_1 = PAB(feature_channels)
        self.block_2 = PAB(feature_channels)
        self.block_3 = PAB(feature_channels)
        self.block_4 = PAB(feature_channels)
        self.block_5 = PAB(feature_channels)
        self.block_6 = PAB(feature_channels)

        self.conv_cat = nn.Conv2d(feature_channels * 4, feature_channels, kernel_size=3, padding=1, bias=True)
        
        self.conv_upsample = nn.Conv2d(feature_channels, out_channels * upscale_height * upscale_width, kernel_size=3, padding=1, stride=1, bias=True)
        self.pixel_shuffle = PixelShuffleCustom(upscale_height, upscale_width)
        
        self.feature_output = feature_output

    def forward(self, x):
        out_feature = self.conv_1(x)
        out_b1, out_b0_2, att1 = self.block_1(out_feature)
        out_b2, out_b1_2, att2 = self.block_2(out_b1)
        out_b3, out_b2_2, att3 = self.block_3(out_b2)
        out_b4, out_b3_2, att4 = self.block_4(out_b3)
        out_b5, out_b4_2, att5 = self.block_5(out_b4)
        out_b6, out_b5_2, att6 = self.block_6(out_b5)

        out = self.conv_cat(torch.cat([out_feature, out_b5, out_b1, out_b6], 1))
        out_feature2 = self.conv_upsample(out)
        output = self.pixel_shuffle(out_feature2)
        
        middle_output = [out_feature]
        middle_output.append(out_b3)
        middle_output.append(out_b6)

        features = None

        if self.feature_output:
            return output, middle_output, features
        else:
            return output