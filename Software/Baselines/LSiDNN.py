import torch
import torch.nn as nn
import torch.nn.functional as F

class LSiDNN(nn.Module):
    def __init__(self, N_fp=108, N_sp=32, N_ch=2, N_f=432, N_s=128):
        super(LSiDNN, self).__init__()
        
        self.N_fp = N_fp
        self.N_sp = N_sp
        self.N_ch = N_ch
        self.in_features = self.N_fp * self.N_sp * self.N_ch
        
        self.N_f = N_f
        self.N_s = N_s
        self.out_features = self.N_f * self.N_s * self.N_ch
        
        self.layer_1 = nn.Linear(in_features=self.in_features, out_features=48)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(in_features=48, out_features=self.out_features)

    def forward(self, x):
        x = x.view(-1, self.in_features)
        out_1 = self.layer_1(x)
        out_1 = self.relu_1(out_1)
        out_final = self.layer_2(out_1)
        out_final = out_final.view(-1, self.N_ch, self.N_f, self.N_s)
        return out_final

if __name__ == '__main__':
    input_freq = 108
    input_spatial = 32
    input_chan = 2
    output_freq = 432
    output_spatial = 128
    
    x = torch.randn(1, input_chan, input_freq, input_spatial)
    model = LSiDNN(N_fp=input_freq, N_sp=input_spatial, N_ch=input_chan, N_f=output_freq, N_s=output_spatial)
    
    x_out = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {x_out.shape}')
    print(f'Frequency: {input_freq} -> {output_freq}')
    print(f'Spatial: {input_spatial} -> {output_spatial}')