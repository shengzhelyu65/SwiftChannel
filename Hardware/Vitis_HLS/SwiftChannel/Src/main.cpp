//Project headers
#include "wrapper.hpp"
#include "model_quan_weights.hpp"
#include "tx_grid.hpp"

// *************************************************
// Sigmoid Activation Stream
// *************************************************
float sigmoid_fcn_float(float input) {
#pragma HLS INLINE
    return 1.0 / (1 + hls::exp(-input)) - 0.5;
}

template <int n_table>
void init_sigmoid_table(F32 table_out[n_table]) {
SIGMOID_INIT_TABLE_LOOP:
    for (int ii = 0; ii < n_table; ii++) {
        // range -8 to +8
        float in_val = 2 * (float) RANGE_SIGMOID * (ii - float(n_table) / 2.0) / float(n_table);
        float real_val = sigmoid_fcn_float(in_val);
        table_out[ii] = (F32) real_val;
    }
}

// *************************************************
// RX Grid
// *************************************************
void init_tx_grid(F32 tx_grid_table[INPUT_HEIGHT][TX_LENGTH][NR_OF_INPUT_CHANNELS]) {
    for (int i = 0; i < INPUT_HEIGHT; i++) {
        for (int j = 0; j < TX_LENGTH; j++) {
            for (int k = 0; k < NR_OF_INPUT_CHANNELS; k++) {
                tx_grid_table[i][j][k] = tx_grid[i][j][k];
            }
        }
    }
}

///-----------------------------------------------------------------------------------------
/// ReLU STREAM
///----------------------------------------------------------------------------------------
template <int nr_iterations>
void relu_stream(stream_u8& stream_in, stream_u8& stream_out, U8 previous_zero_point)
{
    U8 tmp_data;

RELU_LOOP:
    for (int i = 0; i < nr_iterations; i++)
    {
        stream_in.read(tmp_data);
        if (tmp_data < previous_zero_point)
            stream_out.write((U8) previous_zero_point);
        else
            stream_out.write(tmp_data);
    }
}

///-----------------------------------------------------------------------------------------
/// SIGMOID STREAM
///----------------------------------------------------------------------------------------
void sigmoid_stream(stream_u8& stream_in, stream_u8& stream_out, stream_u8& temp_mem,
                        F32 sigmoid_table[NR_OF_TABLE_SIGMOID], 
                        F32 input_scale, U8 input_zero_point,
                        F32 output_scale, U8 output_zero_point,
                        F32 x_scale, U8 x_zero_point
)
{
    I8 tmp_int_in, tmp_int_x, tmp_int_sum;
    F32 tmp_fp_in, tmp_fp_x, tmp_fp_sum, tmp_fp_att;
    U8 tmp_data, tmp_data_x, tmp_sum, result;
    I16 index, tmp_int_16;

    const U16 half_table_size = NR_OF_TABLE_SIGMOID / 2;

SIGMOID_LOOP:
    for (int i = 0; i < FEATURE_SIZE; i++)
    {
        stream_in.read(tmp_data);
        tmp_int_in = tmp_data - input_zero_point;
        tmp_fp_in = tmp_int_in * input_scale;

        temp_mem.read(tmp_data_x);
        tmp_int_x = tmp_data_x - x_zero_point;
        tmp_fp_x = tmp_int_x * x_scale;

        tmp_fp_sum = tmp_fp_in + tmp_fp_x;
        tmp_int_16 = tmp_fp_sum / output_scale + (F32) 0.5 + output_zero_point;
        result = (tmp_int_16 > 255) ? 255 : ((tmp_int_16 < 0) ? 0 : tmp_int_16);
        tmp_int_sum = result - output_zero_point;
        tmp_fp_sum = tmp_int_sum * output_scale;

        index = tmp_fp_in * RANGE_SIGMOID_DIV;
        index = half_table_size + index;
        index = (index < 0) ? 0 : ((index >= NR_OF_TABLE_SIGMOID) ? NR_OF_TABLE_SIGMOID - 1 : index);
        tmp_fp_att = sigmoid_table[index];
        
        result = (tmp_fp_sum * tmp_fp_att) / output_scale + (F32) 0.5 + output_zero_point;

        stream_out.write(result);
    }
}

///-----------------------------------------------------------------------------------------
/// CONVOLUTION LAYER
///----------------------------------------------------------------------------------------
template <int nr_input_channel, int num_iterations>
void Window3D(stream_u8 &pixel_stream, stream_window &window_stream, U8 input_zero_point)
{
    U8 LineBuffer[FILTER_SIZE1_3X3-1][INPUT_WIDTH][nr_input_channel];
#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 type=complete

    window_u8 Window[nr_input_channel];
    window_i8 OutWindow;

    U8 channel_idx = 0;
    U8 col_ptr = 0;
    U8 out_col_ptr = 0;
    U8 out_row_ptr = 0;
    U32 ramp_up = (INPUT_WIDTH+1) * nr_input_channel;
    U32 num_pixels = INPUT_HEIGHT * INPUT_WIDTH * nr_input_channel;

UPDATE_WINDOW_LOOP:
    for (int n = 0; n < num_iterations; n++)
    {
#pragma HLS LOOP_TRIPCOUNT max=num_iterations
#pragma HLS PIPELINE II=1

        U8 new_pixel = (n < num_pixels) ? pixel_stream.read() : (U8) 0;

SHIFT_WINDOW_LOOP:
        for (int i = 0; i < FILTER_SIZE1_3X3; i++) {
            for (int j = 0; j < FILTER_SIZE1_3X3-1; j++) {
                Window[channel_idx].pix[i][j] = Window[channel_idx].pix[i][j+1];
            }
            Window[channel_idx].pix[i][FILTER_SIZE1_3X3-1] = (i < FILTER_SIZE1_3X3-1) ? LineBuffer[i][col_ptr][channel_idx] : new_pixel;
        }

        LineBuffer[0][col_ptr][channel_idx] = LineBuffer[1][col_ptr][channel_idx];
        LineBuffer[1][col_ptr][channel_idx] = new_pixel;
        
        if (n >= ramp_up) {
            for (int i = 0; i < FILTER_SIZE1_3X3; i++) {
                for (int j = 0; j < FILTER_SIZE1_3X3; j++) {
                    I16 xoffset = i + out_row_ptr - 1;
                    I16 yoffset = j + out_col_ptr - 1;
                    OutWindow.pix[i][j] = (yoffset < 0 || yoffset >= INPUT_WIDTH || xoffset < 0 || xoffset >= INPUT_HEIGHT) ? (I8) 0 : (Window[channel_idx].pix[i][j] - input_zero_point);
                }
            }

            out_col_ptr = (channel_idx == nr_input_channel-1) ? ((out_col_ptr == INPUT_WIDTH-1) ? 0 : out_col_ptr+1) : out_col_ptr;
            out_row_ptr = (channel_idx == nr_input_channel-1) ? ((out_col_ptr == 0) ? out_row_ptr+1 : out_row_ptr) : out_row_ptr;
            window_stream.write(OutWindow);
        }

        col_ptr = (channel_idx == nr_input_channel-1) ? ((col_ptr == INPUT_WIDTH-1) ? 0 : col_ptr+1) : col_ptr;
        channel_idx = (channel_idx == nr_input_channel-1) ? 0 : channel_idx+1;
    }
}

template <U8 nr_of_input_channels, U8 nr_of_output_channels, U8 Tn, U8 Tm>
void Filter3D(
    I8              weights[nr_of_input_channels][FILTER_SIZE1_3X3][FILTER_SIZE1_3X3][nr_of_output_channels],
    I32             biases[nr_of_output_channels],
    F32             weights_scale[nr_of_output_channels],
    F32             input_scale,
    F32             output_scale,
    U8              output_zero_point,
    stream_window   &window_stream,
    stream_u8       &pixel_stream)
{
#pragma HLS BIND_STORAGE variable=weights type=rom_np impl=bram
#pragma HLS BIND_STORAGE variable=biases type=rom_np impl=bram
#pragma HLS BIND_STORAGE variable=weights_scale type=rom_np impl=bram

#pragma HLS ARRAY_PARTITION variable=weights dim=2 type=complete
#pragma HLS ARRAY_PARTITION variable=weights dim=3 type=complete

    window_i8 window[nr_of_input_channels];
#pragma HLS ARRAY_PARTITION variable=window dim=1 type=complete

    I16 tmp_int_16;
    U8 result;

    I32 sum[nr_of_output_channels];
    I32 partial_sum[Tm];
#pragma HLS ARRAY_PARTITION variable=partial_sum dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=sum dim=1 type=complete

FILTER_ROW_LOOP:
    for (int y = 0; y < INPUT_HEIGHT; y++) 
    {
FILTER_COL_LOOP:
        for (int x = 0; x < INPUT_WIDTH; x++) 
        {
#pragma HLS PIPELINE off

FILTER_READ_LOOP:
            for (int in_ch = 0; in_ch < nr_of_input_channels; in_ch++) 
            {
                window_stream.read(window[in_ch]);
            }

FILTER_SUM_LOOP:
            for (int out_ch = 0; out_ch < nr_of_output_channels; out_ch++) 
            {
#pragma HLS UNROLL
                sum[out_ch] = biases[out_ch];
            }

FILTER_CAL_OUT_LOOP:
            for (int to = 0; to < nr_of_output_channels; to += Tm) 
            {
#pragma HLS UNROLL off
FILTER_CAL_IN_LOOP:
                for (int ti = 0; ti < nr_of_input_channels; ti += Tn) 
                {
#pragma HLS UNROLL off
#pragma HLS PIPELINE
                    for (int too = 0; too < Tm; too++) 
                    {
#pragma HLS UNROLL
                        partial_sum[too] = 0;
                    }

                    for (int row = 0; row < FILTER_SIZE1_3X3; row++) 
                    {
#pragma HLS UNROLL
                        for (int col = 0; col < FILTER_SIZE1_3X3; col++) 
                        {
#pragma HLS UNROLL
                            for (int too = 0; too < Tm; too++) 
                            {
#pragma HLS UNROLL
                                for (int tii = 0; tii < Tn; tii++) 
                                {
#pragma HLS UNROLL
                                    partial_sum[too] += window[ti + tii].pix[row][col] * weights[ti + tii][row][col][to + too];
                                }
                            }
                        }
                    }

                    for (int too = 0; too < Tm; too++) 
                    {
#pragma HLS UNROLL
                        sum[to + too] += partial_sum[too];
                    }
                }
            }

FILTER_WRITE_LOOP:
            for (int out_ch = 0; out_ch < nr_of_output_channels; out_ch++) 
            {
                tmp_int_16 = sum[out_ch] * input_scale * weights_scale[out_ch] / output_scale + (F32) 0.5 + output_zero_point;
                result = (tmp_int_16 > 255) ? 255 : ((tmp_int_16 < 0) ? 0 : tmp_int_16);
                pixel_stream.write(result);
            }
        }
    }
}

///-----------------------------------------------------------------------------------------
/// UPSAMPLE STREAM
///----------------------------------------------------------------------------------------
template <U8 nr_of_input_channels, U8 Tc>
void Filter3DUpsample(
    I8              weights[nr_of_input_channels][UPSAMPLE_CHANNELS],
    I32             biases[UPSAMPLE_CHANNELS],
    F32             weights_scale[UPSAMPLE_CHANNELS],
    F32             input_scale,
    F32             output_scale,
    U8              input_zero_point,
    U8              output_zero_point,
    stream_u8       &pixel_stream_in,
    stream_u8       &pixel_stream_out)
{
#pragma HLS BIND_STORAGE variable=weights type=rom_np impl=bram
#pragma HLS BIND_STORAGE variable=biases type=rom_np impl=bram
#pragma HLS BIND_STORAGE variable=weights_scale type=rom_np impl=bram

    U8 tmp_pixel;
    I16 tmp_int_16;
    U8 result;

    I8 pixel[Tc][nr_of_input_channels];
    I32 sum[Tc][UPSAMPLE_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=sum dim=0 type=complete
#pragma HLS ARRAY_PARTITION variable=pixel dim=0 type=complete

#pragma HLS ARRAY_PARTITION variable=weights dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=weights dim=2 type=complete

FILTER_ROW_LOOP:
    for (int y = 0; y < INPUT_HEIGHT; y++) 
    {
#pragma HLS UNROLL off
#pragma HLS PIPELINE off
FILTER_COL_LOOP:
        for (int tc = 0; tc < INPUT_WIDTH; tc += Tc)
        {
#pragma HLS UNROLL off
#pragma HLS PIPELINE off

FILTER_READ_LOOP:
            for (int tcc = 0; tcc < Tc; tcc++)
            {
                for (int in_ch = 0; in_ch < nr_of_input_channels; in_ch++) 
                {
#pragma HLS UNROLL off
                    tmp_pixel = pixel_stream_in.read();
                    pixel[tcc][in_ch] = tmp_pixel - input_zero_point;
                }

                for (int out_ch = 0; out_ch < UPSAMPLE_CHANNELS; out_ch++) 
                {
#pragma HLS UNROLL
                    sum[tcc][out_ch] = biases[out_ch];
                }
            }

FILTER_CAL_LOOP:
            for (int tcc = 0; tcc < Tc; tcc++)
            {
#pragma HLS UNROLL
                for (int to = 0; to < UPSAMPLE_CHANNELS; to++) 
                {
#pragma HLS UNROLL
                    for (int ti = 0; ti < nr_of_input_channels; ti++)
                    {
#pragma HLS UNROLL
                        sum[tcc][to] += pixel[tcc][ti] * weights[ti][to];
                    }
                }
            }

FILTER_WRITE_LOOP:
            for (int tcc = 0; tcc < Tc; tcc++)
            {
                for (int out_ch = 0; out_ch < UPSAMPLE_CHANNELS; out_ch++) 
                {
                    tmp_int_16 = sum[tcc][out_ch] * input_scale * weights_scale[out_ch] / output_scale + (F32) 0.5 + output_zero_point;
                    result = (tmp_int_16 > 255) ? 255 : ((tmp_int_16 < 0) ? 0 : tmp_int_16);
                    pixel_stream_out.write(result);
                }
            }
        }
    }
}

///-----------------------------------------------------------------------------------------
/// PIXEL SHUFFLE
///----------------------------------------------------------------------------------------
void pixel_shuffle(stream_u8& stream_in, stream_u8& stream_out)
{
    U8 pix_in;
    U8 pix_out[SCALE_FACTOR][INPUT_WIDTH][SCALE_FACTOR][NR_OF_INPUT_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=pix_out dim=4 type=complete
    U8 new_width;

PIXEL_SHUFFLE_LOOP1:
    for (int current_column = 0; current_column < INPUT_WIDTH; current_column++)
    {
        for (int row_offset = 0; row_offset < SCALE_FACTOR; row_offset++)
        {
            for (int col_offset = 0; col_offset < SCALE_FACTOR; col_offset++)
            {
                stream_in.read(pix_out[row_offset][current_column][col_offset][0]);
            }
        }

        for (int col_offset = 0; col_offset < SCALE_FACTOR; col_offset++)
        {
            stream_out.write(pix_out[0][current_column][col_offset][0]);
            stream_in.read(pix_in);
            stream_out.write(pix_in);
        }

        for (int row_offset = 1; row_offset < SCALE_FACTOR; row_offset++)
        {
            for (int col_offset = 0; col_offset < SCALE_FACTOR; col_offset++)
            {
                stream_in.read(pix_out[row_offset][current_column][col_offset][1]);
            }
        }
    }

PIXEL_SHUFFLE_LOOP2:
    for (int current_line = 1; current_line < INPUT_HEIGHT; current_line++)
    {
#pragma HLS PIPELINE
PIXEL_SHUFFLE_LOOP2_1:
        for (int row_offset = 1; row_offset < SCALE_FACTOR; row_offset++)
        {
            for (int col = 0; col < INPUT_WIDTH; col++)
            {
                for (int col_offset = 0; col_offset < SCALE_FACTOR; col_offset++)
                {
                    for (int channel = 0; channel < NR_OF_INPUT_CHANNELS; channel++)
                    {
                        stream_out.write(pix_out[row_offset][col][col_offset][channel]);
                    }
                }
            }
        }

PIXEL_SHUFFLE_LOOP2_2:
        for (int current_column = 0; current_column < INPUT_WIDTH; current_column++)
        {
            for (int row_offset = 0; row_offset < SCALE_FACTOR; row_offset++)
            {
                for (int col_offset = 0; col_offset < SCALE_FACTOR; col_offset++)
                {
                    stream_in.read(pix_out[row_offset][current_column][col_offset][0]);
                }
            }

            for (int col_offset = 0; col_offset < SCALE_FACTOR; col_offset++)
            {
                stream_out.write(pix_out[0][current_column][col_offset][0]);
                stream_in.read(pix_in);
                stream_out.write(pix_in);
            }

            for (int row_offset = 1; row_offset < SCALE_FACTOR; row_offset++)
            {
                for (int col_offset = 0; col_offset < SCALE_FACTOR; col_offset++)
                {
                    stream_in.read(pix_out[row_offset][current_column][col_offset][1]);
                }
            }
        }
    }

PIXEL_SHUFFLE_LOOP3:
    for (int row_offset = 1; row_offset < SCALE_FACTOR; row_offset++)
    {
        for (int col = 0; col < INPUT_WIDTH; col++)
        {
            for (int col_offset = 0; col_offset < SCALE_FACTOR; col_offset++)
            {
                for (int channel = 0; channel < NR_OF_INPUT_CHANNELS; channel++)
                {
                    stream_out.write(pix_out[row_offset][col][col_offset][channel]);
                }
            }
        }
    }
}

///-----------------------------------------------------------------------------------------
/// SPAB BLOCK
///----------------------------------------------------------------------------------------
void save_temp(stream_u8& stream_in, stream_u8& stream_out, stream_u8& temp_mem)
{
    U8 tmp_data;
    for (int i = 0; i < FEATURE_SIZE; i++)
    {
#pragma HLS PIPELINE II=1
        stream_in.read(tmp_data);
        temp_mem.write(tmp_data);
        stream_out.write(tmp_data);
    }
}

void SPAB(stream_u8& block_in, stream_u8& block_out, 
    I8 weights_block_conv1[FEATURE_CHANNELS][FILTER_SIZE1_3X3][FILTER_SIZE1_3X3][MIDDLE_CHANNELS], I32 biases_block_conv1[MIDDLE_CHANNELS],
    F32 weights_block_conv1_scale[MIDDLE_CHANNELS],
    F32 block_input_scale, U8 block_input_zero_point,
    F32 block_conv1_output_scale, U8 block_conv1_output_zero_point,
    I8 weights_block_conv2[MIDDLE_CHANNELS][FILTER_SIZE1_3X3][FILTER_SIZE1_3X3][FEATURE_CHANNELS], I32 biases_block_conv2[FEATURE_CHANNELS],
    F32 weights_block_conv2_scale[FEATURE_CHANNELS],
    F32 block_conv2_output_scale, U8 block_conv2_output_zero_point,
    F32 sigmoid_table[NR_OF_TABLE_SIGMOID],
    F32 block_output_scale, U8 block_output_zero_point
)
{
#pragma HLS DATAFLOW

    stream_u8 temp_mem("temp_mem");
#pragma HLS STREAM variable=temp_mem depth=FEATURE_SIZE

    stream_u8 block_tmp("block_tmp");
#pragma HLS STREAM variable=block_tmp depth=32
    stream_window block_conv1_window("block_conv1_window");
#pragma HLS STREAM variable=block_conv1_window depth=32
    stream_u8 block_conv1_out("block_conv1_out");
#pragma HLS STREAM variable=block_conv1_out depth=32
    stream_u8 block_conv1_relu("block_conv1_relu");
#pragma HLS STREAM variable=block_conv1_relu depth=32
    stream_window block_conv2_window("block_conv2_window");
#pragma HLS STREAM variable=block_conv2_window depth=32
    stream_u8 block_conv2_out("block_conv2_out");
#pragma HLS STREAM variable=block_conv2_out depth=32

    save_temp(block_in, block_tmp, temp_mem);

    Window3D<FEATURE_CHANNELS, TRIPCOUNT_WIN_FEATURE> (block_tmp, block_conv1_window, block_input_zero_point);

    Filter3D<FEATURE_CHANNELS, MIDDLE_CHANNELS, 4, 4> (weights_block_conv1, biases_block_conv1,
                                                weights_block_conv1_scale, block_input_scale,
                                                block_conv1_output_scale, block_conv1_output_zero_point,
                                                block_conv1_window, block_conv1_out);

    relu_stream<MIDDLE_SIZE> (block_conv1_out, block_conv1_relu, block_conv1_output_zero_point);

    Window3D<MIDDLE_CHANNELS, TRIPCOUNT_WIN_MIDDLE> (block_conv1_relu, block_conv2_window, block_conv1_output_zero_point);

    Filter3D<MIDDLE_CHANNELS, FEATURE_CHANNELS, 4, 4> (weights_block_conv2, biases_block_conv2,
                                                weights_block_conv2_scale, block_conv1_output_scale,
                                                block_conv2_output_scale, block_conv2_output_zero_point,
                                                block_conv2_window, block_conv2_out);

    sigmoid_stream(block_conv2_out, block_out, temp_mem,
                    sigmoid_table,
                    block_conv2_output_scale, block_conv2_output_zero_point,
                    block_output_scale, block_output_zero_point,
                    block_input_scale, block_input_zero_point);
}

///-----------------------------------------------------------------------------------------
/// STREAM INPUT
///----------------------------------------------------------------------------------------
void stream_input(stream_pkt& stream_in, stream_data& stream_out)
{
    transPkt tmp_pkt;
    fp_int tmp_data;
    F32 float_data;

STREAM_INPUT_LOOP:
    for (int i = 0; i < INPUT_RX_SIZE; i++)
    {
        stream_in.read(tmp_pkt);
        tmp_data.i = tmp_pkt.data;
        float_data = (F32) tmp_data.fp;
        stream_out.write(float_data);
    }
}

///-----------------------------------------------------------------------------------------
/// LS ESTIMATOR
///----------------------------------------------------------------------------------------
void ls_estimator(stream_data& stream_in, stream_data& stream_out, F32 tx_grid_table[INPUT_HEIGHT][TX_LENGTH][NR_OF_INPUT_CHANNELS])
{
    F32 rx_float_real, rx_float_imag;
    F32 tx_float_real, tx_float_imag;

LS_EST_FREQ_LOOP:
    for (int freq = 0; freq < INPUT_HEIGHT; freq++)
    {
LS_EST_RX_LOOP:
        for (int rx = 0; rx < RX_LENGTH; rx++)
        {
#pragma HLS PIPELINE II=1
            stream_in.read(rx_float_real);
            stream_in.read(rx_float_imag);

            tx_float_real = tx_grid_table[freq][0][0];
            tx_float_imag = tx_grid_table[freq][0][1];

            stream_out.write(rx_float_real * tx_float_real + rx_float_imag * tx_float_imag);
            stream_out.write(rx_float_imag * tx_float_real - rx_float_real * tx_float_imag);

            tx_float_real = tx_grid_table[freq][1][0];
            tx_float_imag = tx_grid_table[freq][1][1];

            stream_out.write(rx_float_real * tx_float_real + rx_float_imag * tx_float_imag);
            stream_out.write(rx_float_imag * tx_float_real - rx_float_real * tx_float_imag);
        }
    }
}

///-----------------------------------------------------------------------------------------
/// INPUT QUANTIZATION
///----------------------------------------------------------------------------------------
void input_quantization(stream_data& stream_in, stream_u8& stream_out, F32 input_scale, U8 input_zero_point)
{
    F32 tmp_data;
    I16 tmp_data_int;
    U8 quantized_data;

INPUT_QUAN_LOOP:
    for (int i = 0; i < INPUT_GRID_SIZE; i++)
    {
        stream_in.read(tmp_data);
        tmp_data_int = tmp_data / input_scale + (F32) 0.5 + input_zero_point;
        quantized_data = (tmp_data_int > 255) ? 255 : ((tmp_data_int < 0) ? 0 : tmp_data_int);
        stream_out.write(quantized_data);
    }
}

///-----------------------------------------------------------------------------------------
/// OUTPUT DEQUANTIZATION
///----------------------------------------------------------------------------------------
void output_dequantization(stream_u8& stream_in, stream_data& stream_out, F32 output_scale, U8 output_zero_point)
{
    U8 tmp_data;
    I8 tmp_data_int;
    F32 dequantized_data;

OUTPUT_DEQUAN_LOOP:
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        stream_in.read(tmp_data);
        tmp_data_int = tmp_data - output_zero_point;
        dequantized_data = tmp_data_int * output_scale;
        stream_out.write(dequantized_data);
    }
}

///-----------------------------------------------------------------------------------------
/// STREAM OUTPUT
///----------------------------------------------------------------------------------------
void stream_output(stream_data& stream_in, stream_pkt& stream_out)
{
transPkt tmp_pkt;
fp_int tmp_data;
F32 tmp_float;

STREAM_OUTPUT_LOOP:
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        stream_in.read(tmp_float);
        tmp_data.fp = (float) tmp_float;
        tmp_pkt.data = tmp_data.i;
        tmp_pkt.last = (i == OUTPUT_SIZE - 1) ? 1 : 0;
        tmp_pkt.keep = -1;
        tmp_pkt.strb = -1;
        stream_out.write(tmp_pkt);
    }
}

///-----------------------------------------------------------------------------------------
/// SwiftChannel
///----------------------------------------------------------------------------------------
void SwiftChannel(stream_pkt &stream_in, stream_pkt &stream_out)
{
#pragma HLS INTERFACE mode=axis register_mode=both port=stream_in
#pragma HLS INTERFACE mode=axis register_mode=both port=stream_out
#pragma HLS INTERFACE mode=s_axilite port=return

    static F32 sigmoid_table[NR_OF_TABLE_SIGMOID];
#pragma HLS BIND_STORAGE variable=sigmoid_table type=ram_1wnr impl=bram
    
    static F32 tx_grid_table[INPUT_HEIGHT][TX_LENGTH][NR_OF_INPUT_CHANNELS];
#pragma HLS BIND_STORAGE variable=tx_grid_table type=rom_np impl=bram
    
    stream_data stream_in_data("stream_in_data");
#pragma HLS STREAM variable=stream_in_data depth=32
    stream_data ls_estimation("ls_estimation");
#pragma HLS STREAM variable=ls_estimation depth=32
    stream_u8 stream_in_quantized("stream_in_quantized");
#pragma HLS STREAM variable=stream_in_quantized depth=32

    stream_window conv1_window("conv1_window");
#pragma HLS STREAM variable=conv1_window depth=32
    stream_u8 conv1_out("conv1_out");
#pragma HLS STREAM variable=conv1_out depth=32

    stream_u8 block1_out("block1_out");
#pragma HLS STREAM variable=block1_out depth=32
    stream_u8 block2_out("block2_out");
#pragma HLS STREAM variable=block2_out depth=32
    stream_u8 block3_out("block3_out");
#pragma HLS STREAM variable=block3_out depth=32
    stream_u8 block4_out("block4_out");
#pragma HLS STREAM variable=block4_out depth=32

    stream_window conv2_window("conv2_window");
#pragma HLS STREAM variable=conv2_window depth=32
    stream_u8 conv2_out("conv2_out");
#pragma HLS STREAM variable=conv2_out depth=32

    stream_window upsample_window("upsample_window");
#pragma HLS STREAM variable=upsample_window depth=32
    stream_u8 upsample_out("upsample_out");
#pragma HLS STREAM variable=upsample_out depth=32

    stream_u8 shuffle_out("shuffle_out");
#pragma HLS STREAM variable=shuffle_out depth=32
    stream_data stream_out_dequantized("stream_out_dequantized");
#pragma HLS STREAM variable=stream_out_dequantized depth=32

    init_sigmoid_table<NR_OF_TABLE_SIGMOID> (sigmoid_table);

    init_tx_grid(tx_grid_table);

#pragma HLS DATAFLOW

    stream_input(stream_in, stream_in_data);

    ls_estimator(stream_in_data, ls_estimation, tx_grid_table);

    input_quantization(ls_estimation, stream_in_quantized, input_scale, input_zero_point);

    Window3D<NR_OF_INPUT_CHANNELS, TRIPCOUNT_WIN_INPUT> (stream_in_quantized, conv1_window, input_zero_point);
    
    Filter3D<NR_OF_INPUT_CHANNELS, FEATURE_CHANNELS, 2, 4> (weights_conv1, bias_conv1,
                                                    weights_conv1_scale, input_scale,
                                                    output_conv1_scale, output_conv1_zero_point,
                                                    conv1_window, conv1_out);
    
    SPAB(conv1_out, block1_out, 
        weights_block1_conv1, bias_block1_conv1,
        weights_block1_conv1_scale,
        output_conv1_scale, output_conv1_zero_point, 
        output_block1_conv1_scale, output_block1_conv1_zero_point,
        weights_block1_conv2, bias_block1_conv2,
        weights_block1_conv2_scale,
        output_block1_conv2_scale, output_block1_conv2_zero_point,
        sigmoid_table,
        output_block1_scale, output_block1_zero_point
    );

    SPAB(block1_out, block2_out,
        weights_block2_conv1, bias_block2_conv1,
        weights_block2_conv1_scale,
        output_block1_scale, output_block1_zero_point,
        output_block2_conv1_scale, output_block2_conv1_zero_point,
        weights_block2_conv2, bias_block2_conv2,
        weights_block2_conv2_scale,
        output_block2_conv2_scale, output_block2_conv2_zero_point,
        sigmoid_table,
        output_block2_scale, output_block2_zero_point
    );

    SPAB(block2_out, block3_out,
        weights_block3_conv1, bias_block3_conv1,
        weights_block3_conv1_scale,
        output_block2_scale, output_block2_zero_point,
        output_block3_conv1_scale, output_block3_conv1_zero_point,
        weights_block3_conv2, bias_block3_conv2,
        weights_block3_conv2_scale,
        output_block3_conv2_scale, output_block3_conv2_zero_point,
        sigmoid_table,
        output_block3_scale, output_block3_zero_point
    );

    SPAB(block3_out, block4_out,
        weights_block4_conv1, bias_block4_conv1,
        weights_block4_conv1_scale,
        output_block3_scale, output_block3_zero_point,
        output_block4_conv1_scale, output_block4_conv1_zero_point,
        weights_block4_conv2, bias_block4_conv2,
        weights_block4_conv2_scale,
        output_block4_conv2_scale, output_block4_conv2_zero_point,
        sigmoid_table,
        output_block4_scale, output_block4_zero_point
    );

    Window3D<FEATURE_CHANNELS, TRIPCOUNT_WIN_FEATURE> (block4_out, conv2_window, output_block4_zero_point);
    
    Filter3D<FEATURE_CHANNELS, DOWNSAMPLE_CHANNELS, 4, 4> (weights_conv2, bias_conv2,
                                                    weights_conv2_scale, output_block4_scale,
                                                    output_conv2_scale, output_conv2_zero_point,
                                                    conv2_window, conv2_out);

    Filter3DUpsample<DOWNSAMPLE_CHANNELS, 2> (weights_conv_upsample, bias_conv_upsample,
                    weights_conv_upsample_scale, 
                    output_conv2_scale, output_conv_upsample_scale,
                    output_conv2_zero_point, output_conv_upsample_zero_point,
                    conv2_out, upsample_out);

    pixel_shuffle(upsample_out, shuffle_out);
    
    output_dequantization(shuffle_out, stream_out_dequantized, output_conv_upsample_scale, output_conv_upsample_zero_point);

    stream_output(stream_out_dequantized, stream_out);
}