import torch
import numpy as np
import warnings
from Models.swiftchannel_student_quan import SwiftChannelQuan

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='ignore',
    category=FutureWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

def numpy_to_cpp_static_matrix(name, array, size, cpp_type):
    if len(size) == 1:
        cpp_string_0 = f"static {cpp_type} {name}[{size[0]}] = \n"
    elif len(size) == 2:
        cpp_string_0 = f"static {cpp_type} {name}[{size[0]}][{size[1]}] = \n"
    elif len(size) == 3:
        cpp_string_0 = f"static {cpp_type} {name}[{size[0]}][{size[1]}][{size[2]}] = \n"
    elif len(size) == 4:
        cpp_string_0 = f"static {cpp_type} {name}[{size[0]}][{size[1]}][{size[2]}][{size[3]}] = \n"
        
    if len(size) == 4 and size[1] == 1 and size[2] == 1:
        cpp_string_0 = f"static {cpp_type} {name}[{size[0]}][{size[3]}] = \n"
        array = array.reshape((size[0], size[3]))

    # Set print options for formatting
    if cpp_type == 'U8' or cpp_type == 'I8' or cpp_type == 'I32':
        np.set_printoptions(threshold=np.inf, formatter={'int_kind': lambda x: f"{x:d}"})
    elif cpp_type == 'F32':
        np.set_printoptions(threshold=np.inf, formatter={'float_kind': lambda x: f"{x:.15f}"})
    cpp_string = np.array2string(array, separator=', ')
    cpp_string = cpp_string.replace('[', '{').replace(']', '}')
    cpp_string += ";\n"
    return cpp_string_0 + cpp_string

def process_conv_layer(layer, input_scale, num_output_channels, block_name='conv1'):
    global uint8_str, int8_str, int32_str, float_str
    
    total_str = ''
    
    # Extract output scale and zero point
    output_scale = float(layer.scale)
    output_zero_point = int(layer.zero_point)
    output_scale_cpp = f"static {float_str} output_{block_name}_scale = {output_scale:.15f};\n"
    output_zero_point_cpp = f"static {uint8_str} output_{block_name}_zero_point = {output_zero_point};\n"
    
    # Extract weight scales and zero points
    weight_scale = np.zeros(num_output_channels, dtype=np.float32)
    weight_zero_point = np.zeros(num_output_channels, dtype=np.uint8)
    for i in range(num_output_channels):
        weight_scale[i] = float(layer.weight()[i].q_scale())
        weight_zero_point[i] = int(layer.weight()[i].q_zero_point())
    weight_scale_cpp = numpy_to_cpp_static_matrix(f"weights_{block_name}_scale", weight_scale, weight_scale.shape, float_str)
    weight_zero_point_cpp = numpy_to_cpp_static_matrix(f"weights_{block_name}_zero_point", weight_zero_point, weight_zero_point.shape, uint8_str)
    
    # Extract int-representation of weights and reshape
    weights = layer.weight()
    weights_int = torch.int_repr(weights).numpy().transpose(1, 2, 3, 0)
    # weights_int = weights_int.reshape((weights_int.shape[0], weights_int.shape[1] * weights_int.shape[2], weights_int.shape[3]))
    for i in range(num_output_channels):
        weights_int[:, :, :, i] = weights_int[:, :, :, i] - weight_zero_point[i]
    weights_int_cpp = numpy_to_cpp_static_matrix(f"weights_{block_name}", weights_int, weights_int.shape, int8_str)
    
    # Extract and compute bias in int form
    bias = layer.bias().detach().numpy()
    bias_int = np.zeros_like(bias, dtype=np.int32)
    for i in range(num_output_channels):
        bias_int[i] = np.round(bias[i] / (input_scale * weight_scale[i]))
    bias_int_cpp = numpy_to_cpp_static_matrix(f"bias_{block_name}", bias_int, bias_int.shape, int32_str)
    
    total_str += output_scale_cpp + '\n'
    total_str += output_zero_point_cpp + '\n'
    total_str += weight_scale_cpp + '\n'
    total_str += weight_zero_point_cpp + '\n'
    total_str += weights_int_cpp + '\n'
    total_str += bias_int_cpp + '\n'
    
    return total_str, output_scale, output_zero_point

def process_sigmoid(layer, previous_scale, previous_zero_point, num_output_channels, input_height=108, input_width=32):
    global float_str, uint8_str
    
    total_str = ''
    
    x_ref = torch.randn(1, num_output_channels, input_height, input_width)
    x_ref_quan = torch.quantize_per_tensor(x_ref, scale=previous_scale, zero_point=previous_zero_point, dtype=torch.quint8)
    
    act_output = layer(x_ref_quan)
    sigmoid_scale = act_output.q_scale()
    sigmoid_output_zero_point = act_output.q_zero_point()
    sigmpoid_output_scale_cpp = f"static {float_str} output_sigmoid_scale = {float(sigmoid_scale):.15f};\n"
    sigmoid_output_zero_point_cpp = f"static {uint8_str} output_sigmoid_zero_point = {int(sigmoid_output_zero_point)};\n"
    print("Sigmoid scale: ", sigmoid_scale)
    print("Sigmoid zero point: ", sigmoid_output_zero_point)
    
    total_str += sigmpoid_output_scale_cpp + '\n'
    total_str += sigmoid_output_zero_point_cpp + '\n'
    
    return total_str, sigmoid_scale, sigmoid_output_zero_point

def process_block_output(layer, previous_scale, previous_zero_point, num_output_channels, block_name='block1', input_height=108, input_width=32):
    global float_str, uint8_str
    
    total_str = ''
    
    x_ref = torch.randn(1, num_output_channels, input_height, input_width)
    x_ref_quan = torch.quantize_per_tensor(x_ref, scale=previous_scale, zero_point=previous_zero_point, dtype=torch.quint8)
    
    output_forward = layer.forward(x_ref_quan)[0]
    output_scale = output_forward.q_scale()
    output_zero_point = output_forward.q_zero_point()
    output_scale_cpp = f"static {float_str} output_{block_name}_scale = {float(output_scale):.15f};\n"
    output_zero_point_cpp = f"static {uint8_str} output_{block_name}_zero_point = {int(output_zero_point)};\n"
    print(f"Output scale of {block_name}: ", output_scale)
    print(f"Output zero point of {block_name}: ", output_zero_point)
    
    output_sum = layer.forward(x_ref_quan)[1]
    output_sum_scale = output_sum.q_scale()
    output_sum_zero_point = output_sum.q_zero_point()
    output_sum_scale_cpp = f"static {float_str} output_{block_name}_sum_scale = {float(output_sum_scale):.15f};\n"
    output_sum_zero_point_cpp = f"static {uint8_str} output_{block_name}_sum_zero_point = {int(output_sum_zero_point)};\n"
    print(f"Output sum scale of {block_name}: ", output_sum_scale)
    print(f"Output sum zero point of {block_name}: ", output_sum_zero_point)
    
    total_str += output_scale_cpp + '\n'
    total_str += output_zero_point_cpp + '\n'
    total_str += output_sum_scale_cpp + '\n'
    total_str += output_sum_zero_point_cpp + '\n'
    
    return total_str, output_scale, output_zero_point

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(
        action='ignore',
        category=DeprecationWarning,
        module=r'.*'
    )
    warnings.filterwarnings(
        action='default',
        module=r'torch.ao.quantization'
    )

    uint8_str = "U8"
    int8_str = "I8"
    int32_str = "I32"
    float_str = "F32"
    
    date_exp = '0508'
    time_exp = '1430'
    model_exp = 'QAT'
    loss_exp = 'NMSE'
    folder_path = 'Experiments/Experiment_' + date_exp + '_' + time_exp + '_' + model_exp + '_' + loss_exp + '/'

    middle_channels = 8
    feature_channels = 12
    downsample_channels = 4
    
    model = SwiftChannelQuan(2, 2, middle_channels=middle_channels, feature_channels=feature_channels, upscale=4)
    model.train().to('cpu')
    model.qconfig = torch.quantization.get_default_qat_qconfig('x86')
    torch.backends.quantized.engine = 'x86'
    torch.quantization.prepare_qat(model.train(), inplace=True)
    model.eval()
    checkpoint = torch.load(folder_path + 'model_quantized.pth', map_location='cpu', weights_only=True)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    torch.ao.quantization.convert(model, inplace=True)
    model.eval().to('cpu')
    
    ## Input
    input_scale = float(model.quan.scale)
    input_scale_cpp = f"static {float_str} input_scale = {input_scale:.15f};\n"
    input_zero_point = int(model.quan.zero_point)
    input_zero_point_cpp = f"static {uint8_str} input_zero_point = {input_zero_point};\n"
    print("Input scale: ", input_scale)
    print("Input zero point: ", input_zero_point)

    ## Conv1
    conv1_str, output_conv1_scale, output_conv1_zero_point = process_conv_layer(model.conv_1, input_scale, num_output_channels=feature_channels, block_name='conv1')
    
    ## SPAB block
    block1_str = ''
    block1_conv1_str, output_block1_conv1_scale, _ = process_conv_layer(model.block_1.c1_r, output_conv1_scale, num_output_channels=middle_channels, block_name='block1_conv1')
    block1_conv2_str, output_block1_conv2_scale, output_block1_conv2_zero_point = process_conv_layer(model.block_1.c2_r, output_block1_conv1_scale, num_output_channels=feature_channels, block_name='block1_conv2')
    block1_output_str, output_block1_scale, output_block1_zero_point = process_block_output(model.block_1, output_conv1_scale, output_conv1_zero_point, num_output_channels=feature_channels, block_name='block1')
    block1_str += block1_conv1_str + '\n'
    block1_str += block1_conv2_str + '\n'
    block1_str += block1_output_str + '\n'
    
    ## Sigmoid block
    sigmoid_str, output_sigmoid_scale, output_sigmoid_zero_point = process_sigmoid(model.block_1.act2, output_block1_conv2_scale, output_block1_conv2_zero_point, num_output_channels=feature_channels)
    
    ## SPAB block
    block2_str = ''
    block2_conv1_str, output_block2_conv1_scale, _ = process_conv_layer(model.block_2.c1_r, output_block1_scale, num_output_channels=middle_channels, block_name='block2_conv1')
    block2_conv2_str, _, _ = process_conv_layer(model.block_2.c2_r, output_block2_conv1_scale, num_output_channels=feature_channels, block_name='block2_conv2')
    block2_output_str, output_block2_scale, output_block2_zero_point = process_block_output(model.block_2, output_block1_scale, output_block1_zero_point, num_output_channels=feature_channels, block_name='block2')
    block2_str += block2_conv1_str + '\n'
    block2_str += block2_conv2_str + '\n'
    block2_str += block2_output_str + '\n'
    
    ## SPAB block
    block3_str = ''
    block3_conv1_str, output_block3_conv1_scale, _ = process_conv_layer(model.block_3.c1_r, output_block2_scale, num_output_channels=middle_channels, block_name='block3_conv1')
    block3_conv2_str, _, _ = process_conv_layer(model.block_3.c2_r, output_block3_conv1_scale, num_output_channels=feature_channels, block_name='block3_conv2')
    block3_output_str, output_block3_scale, output_block3_zero_point = process_block_output(model.block_3, output_block2_scale, output_block2_zero_point, num_output_channels=feature_channels, block_name='block3')
    block3_str += block3_conv1_str + '\n'
    block3_str += block3_conv2_str + '\n'
    block3_str += block3_output_str + '\n'
    
    ## SPAB block
    block4_str = ''
    block4_conv1_str, output_block4_conv1_scale, _ = process_conv_layer(model.block_4.c1_r, output_block3_scale, num_output_channels=middle_channels, block_name='block4_conv1')
    block4_conv2_str, _, _ = process_conv_layer(model.block_4.c2_r, output_block4_conv1_scale, num_output_channels=feature_channels, block_name='block4_conv2')
    block4_output_str, output_block4_scale, output_block4_zero_point = process_block_output(model.block_4, output_block3_scale, output_block3_zero_point, num_output_channels=feature_channels, block_name='block4')
    block4_str += block4_conv1_str + '\n'
    block4_str += block4_conv2_str + '\n'
    block4_str += block4_output_str + '\n'
    
    ## Conv 2
    conv2_str, output_conv2_scale, output_conv2_zero_point = process_conv_layer(model.conv_2, output_block4_scale, num_output_channels=downsample_channels, block_name='conv2')
    
    ## Conv upsample
    conv_upsample_str, output_conv_upsample_scale, output_conv_upsample_zero_point = process_conv_layer(model.conv_upsample, output_conv2_scale, num_output_channels=32, block_name='conv_upsample')

    with open('Outputs/model_quan_weights.hpp', 'w') as f:
        f.write(input_scale_cpp)
        f.write('\n')
        f.write(input_zero_point_cpp)
        f.write('\n')
        f.write(conv1_str)
        f.write('\n')
        f.write(block1_str)
        f.write('\n')
        f.write(sigmoid_str)
        f.write('\n')
        f.write(block2_str)
        f.write('\n')
        f.write(block3_str)
        f.write('\n')
        f.write(block4_str)
        f.write('\n')
        f.write(conv2_str)
        f.write('\n')
        f.write(conv_upsample_str)
        f.close()
        