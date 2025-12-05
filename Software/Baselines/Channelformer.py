import torch
import torch.nn as nn
from torch import Tensor
import math
import numpy as np
    
class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: Tensor, mask: Tensor = None):
        batch_size = key.size(0)
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_key).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.d_key).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.d_key).permute(0, 2, 1, 3)
        scaled_dot_prod = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.d_key)
        if mask is not None:
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask == 0, -1e10)
        attn_probs = torch.softmax(scaled_dot_prod, dim=-1)
        A = torch.matmul(self.dropout(attn_probs), V)
        A = A.permute(0, 2, 1, 3).contiguous()
        A = A.view(batch_size, -1, self.n_heads*self.d_key)
        output = self.Wo(A)
        return output, attn_probs

class PreNetwork(nn.Module):
    def __init__(self, input_freq=156, input_spatial=16, input_chan=2, N_filters_enc=5):
        super(PreNetwork, self).__init__()
        self.input_freq = input_freq
        self.input_spatial = input_spatial
        self.input_chan = input_chan
        self.input_dim = self.input_chan * self.input_freq * self.input_spatial
        
        self.conv1 = nn.Conv2d(1, N_filters_enc, kernel_size=(3, 3), padding=(1, 1))
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(N_filters_enc, 1, kernel_size=(3, 3), padding=(1, 1))
        self.ln = nn.LayerNorm(self.input_dim)
        
    def forward(self, x):
        x_input = x.view(-1, 1, self.input_chan, self.input_freq * self.input_spatial)
        x_conv1 = self.conv1(x_input)
        x_gelu = self.gelu(x_conv1)
        x_conv2 = self.conv2(x_gelu)
        x_out = x_conv2.view(-1, self.input_dim)
        x_add = x + x_out
        x_ln = self.ln(x_add)
        return x_ln
    
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads=2, dropout=0, input_freq=156, input_spatial=16, input_chan=2, N_filters_enc=5):
        super(Encoder, self).__init__()
        self.input_freq = input_freq
        self.input_spatial = input_spatial
        self.input_chan = input_chan
        self.input_dim = self.input_chan * self.input_freq * self.input_spatial
        
        self.fc1 = nn.Linear(self.input_dim, 3 * self.input_dim)
        self.mhsa = MultiheadAttention(d_model, n_heads, dropout)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim)
        self.norm1 = nn.LayerNorm(self.input_dim)
        
        self.pre_net = PreNetwork(input_freq=input_freq, input_spatial=input_spatial, input_chan=input_chan, N_filters_enc=N_filters_enc)
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x_fc1 = self.fc1(x)
        split_size = x_fc1.size(-1) // 3
        Q, K, V = torch.split(x_fc1, split_size, dim=-1)
        x_mhsa, _ = self.mhsa(Q, K, V)
        x_mhsa = x_mhsa.squeeze(1)
        x_fc2 = self.fc2(x_mhsa)
        x = x + x_fc2
        x = self.norm1(x)
        x_pre = self.pre_net(x)
        return x_pre
    
class ResidualConv(nn.Module):
    def __init__(self, N_filters_dec, kernel_size=5):
        super(ResidualConv, self).__init__()
        self.conv_1 = nn.Conv2d(N_filters_dec, N_filters_dec, kernel_size=(kernel_size, kernel_size), padding=(kernel_size//2, kernel_size//2))
        self.conv_2 = nn.Conv2d(N_filters_dec, N_filters_dec, kernel_size=(kernel_size, kernel_size), padding=(kernel_size//2, kernel_size//2))
        
    def forward(self, x):
        x_conv1 = self.conv_1(x)
        x_relu = torch.relu(x_conv1)
        x_conv2 = self.conv_2(x_relu)
        x_out = x_conv2 + x
        return x_out
    
class Upsampling(nn.Module):
    def __init__(self, upscale_factor_freq, upscale_factor_spatial, N_filters_dec, kernel_size, input_freq=156, input_spatial=16, input_chan=2):
        super(Upsampling, self).__init__()
        self.input_freq = input_freq
        self.input_spatial = input_spatial
        self.input_chan = input_chan
        self.input_dim = self.input_chan * self.input_freq * self.input_spatial
        
        self.upscale_factor_freq = upscale_factor_freq
        self.upscale_factor_spatial = upscale_factor_spatial
        self.total_upscale = upscale_factor_freq * upscale_factor_spatial
        
        self.fc = nn.Linear(self.input_dim, self.total_upscale * self.input_dim)
        self.conv = nn.Conv2d(N_filters_dec, 1, kernel_size=(kernel_size, kernel_size), padding=(kernel_size//2, kernel_size//2))
        
        self.N_filters_dec = N_filters_dec
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x_fc = self.fc(x)
        x_fc = x_fc.view(-1, self.N_filters_dec, self.input_chan, 
                         self.input_freq * self.upscale_factor_freq * self.input_spatial * self.upscale_factor_spatial)
        
        x_conv = self.conv(x_fc)
        x_out = x_conv.view(-1, self.input_chan, 
                            self.input_freq * self.upscale_factor_freq, 
                            self.input_spatial * self.upscale_factor_spatial)
        return x_out
        
class Decoder(nn.Module):
    def __init__(self, N_filters_dec=12, kernel_size=5, N_res_blocks=3, upscale_factor_freq=4, upscale_factor_spatial=4, input_freq=156, input_spatial=16, input_chan=2):
        super(Decoder, self).__init__()
        self.input_freq = input_freq
        self.input_spatial = input_spatial
        self.input_chan = input_chan
        
        self.cnn_1 = nn.Conv2d(1, N_filters_dec, kernel_size=(kernel_size, kernel_size), padding=(kernel_size//2, kernel_size//2))
        self.res_blocks = nn.ModuleList([ResidualConv(N_filters_dec, kernel_size) for _ in range(N_res_blocks)])
        self.upsampling = Upsampling(upscale_factor_freq, upscale_factor_spatial, N_filters_dec, kernel_size, input_freq, input_spatial, input_chan)
        
    def forward(self, x):
        x = x.view(-1, 1, self.input_chan, self.input_freq * self.input_spatial)        
        x_cnn = self.cnn_1(x)
        
        for res_block in self.res_blocks:
            x_cnn = res_block(x_cnn)
        
        x_upsample = self.upsampling(x_cnn)
        
        return x_upsample
    
class Channelformer(nn.Module):
    def __init__(self, input_freq=108, input_spatial=32, input_chan=2, N_filters_enc=5, N_filters_dec=12, kernel_size=5, N_res_blocks=3, upscale_factor_freq=4, upscale_factor_spatial=4):
        super(Channelformer, self).__init__()
        self.encoder = Encoder(d_model=(input_freq*input_spatial*input_chan), input_freq=input_freq, input_spatial=input_spatial, input_chan=input_chan)
        self.decoder = Decoder(N_filters_dec=N_filters_dec, kernel_size=kernel_size, N_res_blocks=N_res_blocks, 
                               upscale_factor_freq=upscale_factor_freq, upscale_factor_spatial=upscale_factor_spatial,
                               input_freq=input_freq, input_spatial=input_spatial, input_chan=input_chan)
        
    def forward(self, x):
        x_enc = self.encoder(x)
        x_out = self.decoder(x_enc)
        return x_out
        

if __name__ == '__main__':
    input_freq = 108
    input_spatial = 32
    input_chan = 2
    N_filters_enc = 5
    N_filters_dec = 12
    kernel_size = 5
    N_res_blocks = 3
    upscale_factor_freq = 4
    upscale_factor_spatial = 4
    
    x = torch.randn(1, input_chan, input_freq, input_spatial)
    channelformer = Channelformer(input_freq=input_freq, input_spatial=input_spatial, input_chan=input_chan, 
                                 N_filters_enc=N_filters_enc, N_filters_dec=N_filters_dec, 
                                 kernel_size=kernel_size, N_res_blocks=N_res_blocks, 
                                 upscale_factor_freq=upscale_factor_freq, upscale_factor_spatial=upscale_factor_spatial)
    
    x_out = channelformer(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {x_out.shape}')
    print(f'Frequency upscale: {input_freq} -> {input_freq * upscale_factor_freq}')
    print(f'Spatial upscale: {input_spatial} -> {input_spatial * upscale_factor_spatial}')