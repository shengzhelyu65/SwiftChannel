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
    
class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        gain = gain1

        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=True),
        )

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=1, bias=True)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat

    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        return out
    
class CustomAct(nn.Module):
    def __init__(self):
        super(CustomAct, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(x) - 0.5
        return out

class SPAB(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super(SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2)
        self.c2_r = Conv3XC(mid_channels, out_channels, gain1=2)
        self.act1 = nn.ReLU(inplace=False)
        self.act2 = CustomAct()

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        sim_att = self.act2(out2)
        
        out = (x + out2) * sim_att

        return out, out1, sim_att

class SwiftChannelStudent(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(self, num_in_ch, num_out_ch, middle_channels, feature_channels, downsample_channels=4, teacher_channels=24, upscale=4, feature_output=False):
        super(SwiftChannelStudent, self).__init__()
        
        if isinstance(upscale, tuple):
            upscale_height, upscale_width = upscale
        else:
            upscale_height = upscale
            upscale_width = upscale

        in_channels = num_in_ch
        out_channels = num_out_ch

        self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2)
        
        self.block_1 = SPAB(in_channels=feature_channels, mid_channels=middle_channels, out_channels=feature_channels)
        self.block_2 = SPAB(in_channels=feature_channels, mid_channels=middle_channels, out_channels=feature_channels)
        self.block_3 = SPAB(in_channels=feature_channels, mid_channels=middle_channels, out_channels=feature_channels)
        self.block_4 = SPAB(in_channels=feature_channels, mid_channels=middle_channels, out_channels=feature_channels)
        
        self.conv_2 = Conv3XC(feature_channels, downsample_channels, gain1=2)
        
        self.conv_upsample = nn.Conv2d(downsample_channels, out_channels * upscale_height * upscale_width, kernel_size=1, padding=0, stride=1, bias=True)
        self.pixel_shuffle = PixelShuffleCustom(upscale_height, upscale_width)
        
        self.feature_output = feature_output
        # Include an extra regressor (in our case linear)
        self.regressor_1 = nn.Sequential(
            nn.Conv2d(feature_channels, teacher_channels, kernel_size=3, padding=1)
        )
        self.regressor_2 = nn.Sequential(
            nn.Conv2d(feature_channels, teacher_channels, kernel_size=3, padding=1)
        )
        self.regressor_3 = nn.Sequential(
            nn.Conv2d(feature_channels, teacher_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out_feature = self.conv_1(x)
        out_b1, out_b0_2, att1 = self.block_1(out_feature)
        out_b2, out_b1_2, att2 = self.block_2(out_b1)
        out_b3, out_b2_2, att3 = self.block_3(out_b2)
        out_b4, out_b3_2, att4 = self.block_4(out_b3)
        
        out_feature2 = self.conv_2(out_b4)
        
        out_upsample_pre = self.conv_upsample(out_feature2)
        output = self.pixel_shuffle(out_upsample_pre)
        
        output_feature_teacher = self.regressor_1(out_feature)
        output_b2_teacher = self.regressor_2(out_b2)
        output_b4_teacher = self.regressor_3(out_b4)
        middle_output = [output_feature_teacher]
        middle_output.append(output_b2_teacher)
        middle_output.append(output_b4_teacher)

        features = None
        
        if self.feature_output:
            return output, middle_output, features
        else:
            return output