from Baselines.Channelformer import Channelformer
from Baselines.ChannelNet import ChannelNet
from Baselines.FSRCNN import FSRCNN
from Baselines.LSiDNN import LSiDNN
from Baselines.ReEsNet import ReEsNet
from Baselines.I_ResNet import I_ResNet

def return_model(model_selection):
    if model_selection == 'Channelformer':
        model = Channelformer(input_freq=108, input_spatial=32, input_chan=2, 
                             upscale_factor_freq=4, upscale_factor_spatial=4)
    elif model_selection == 'ChannelNet':
        model = ChannelNet(channels=2)
    elif model_selection == 'FSRCNN':
        model = FSRCNN(scale_factor=4, num_channels=2)
    elif model_selection == 'I_ResNet':
        model = I_ResNet(upscale_factor=4, n_filters=8)
    elif model_selection == 'LSiDNN':
        model = LSiDNN(N_fp=108, N_sp=32, N_ch=2, N_f=432, N_s=128)
    elif model_selection == 'ReEsNet':
        model = ReEsNet(upscale_factor=4)
    else:
        raise ValueError(f"Unknown model selection: {model_selection}")
        
    return model