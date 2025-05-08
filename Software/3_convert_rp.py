import torch
import numpy as np
from Models.swiftchannel_student import SwiftChannelStudent
from Models.swiftchannel_student_rp import SwiftChannelRP

def load_checkpoint(model, optimizer=None, scheduler=None, from_checkpoint=None):
    checkpoint = torch.load(from_checkpoint, map_location='cuda', weights_only=True)
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    return model, optimizer, scheduler, epoch

def copy_conv_para(layer_src, layer_dst, original_state_dict):
    eval_conv1_weight = original_state_dict[layer_src +'.eval_conv.weight']
    eval_conv1_bias = original_state_dict[layer_src +'.eval_conv.bias']
    layer_dst.weight.data.copy_(eval_conv1_weight)
    layer_dst.bias.data.copy_(eval_conv1_bias)
    
    assert torch.equal(layer_dst.weight.data, eval_conv1_weight)
    assert torch.equal(layer_dst.bias.data, eval_conv1_bias)

def save_checkpoint(model, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, filename)
    
if __name__ == "__main__":
    original_model = SwiftChannelStudent(2, 2, upscale=4, middle_channels=8, feature_channels=12, feature_output=False)
    rp_model = SwiftChannelRP(2, 2, upscale=4, middle_channels=8, feature_channels=12)
    
    exp_date = '0508'
    exp_time = '1423'
    model_used = 'Distillation'
    loss_used = 'NMSE'
    fre = 'FRE_4'
    spa = 'SPA_4'
    file_save_name = f"Experiments/Experiment_{exp_date}_{exp_time}_{model_used}_{loss_used}_{fre}_{spa}"
    
    original_model, _, _, _ = load_checkpoint(original_model, None, None, f"{file_save_name}/model_checkpoint_best.pth")
    original_model.eval().to('cuda')
    rp_model.train().to('cuda')
    
    original_state_dict = original_model.state_dict()

    # Copy weights and biases into the new model
    copy_conv_para('conv_1', rp_model.conv_1, original_state_dict)
    copy_conv_para('conv_2', rp_model.conv_2, original_state_dict)
    
    copy_conv_para('block_1.c1_r', rp_model.block_1.c1_r, original_state_dict)
    copy_conv_para('block_1.c2_r', rp_model.block_1.c2_r, original_state_dict)
    copy_conv_para('block_2.c1_r', rp_model.block_2.c1_r, original_state_dict)
    copy_conv_para('block_2.c2_r', rp_model.block_2.c2_r, original_state_dict)
    copy_conv_para('block_3.c1_r', rp_model.block_3.c1_r, original_state_dict)
    copy_conv_para('block_3.c2_r', rp_model.block_3.c2_r, original_state_dict)
    copy_conv_para('block_4.c1_r', rp_model.block_4.c1_r, original_state_dict)
    copy_conv_para('block_4.c2_r', rp_model.block_4.c2_r, original_state_dict)
    
    conv_upsample_weight = original_state_dict['conv_upsample.weight']
    conv_upsample_bias = original_state_dict['conv_upsample.bias']
    rp_model.conv_upsample.weight.data.copy_(conv_upsample_weight)
    rp_model.conv_upsample.bias.data.copy_(conv_upsample_bias)

    # Save the new model's state_dict if needed
    rp_model.eval().to('cuda')
    save_checkpoint(rp_model, f"{file_save_name}/model_checkpoint_best_eval.pth")