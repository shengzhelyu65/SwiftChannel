#pragma once
#ifndef WRAPPER_HPP
#define WRAPPER_HPP

///includes
#include <ap_int.h>                                     // Brings the ap_uint type
#include <hls_stream.h>                                 // Brings the stream
#include <ap_axi_sdata.h>
#include <hls_math.h>
#include <ap_fixed.h>

using namespace std;

///---------------------------------------------------------------------------------------------------------------------
///Define useful parameters
///---------------------------------------------------------------------------------------------------------------------
#define NR_OF_INPUT_CHANNELS        2

#define NUM_FFT         			512
#define NUM_SUBCARRIERS             432
#define RX_LENGTH                   16
#define TX_LENGTH                   2
#define SCALE_FACTOR                4
#define SCALE_FACTOR2			   	16

#define INPUT_HEIGHT                108
#define INPUT_WIDTH                 32

#define PAD_HEIGHT                  110                 //** should be INPUT_HEIGHT + 2*PADDING_NUMBER            
#define PAD_WIDTH                   34                  //** should be INPUT_WIDTH + 2*PADDING_NUMBER

#define OUTPUT_HEIGHT               432
#define OUTPUT_WIDTH                128

#define PADDING_NUMBER              1
#define STRIDE                      1

#define MIDDLE_CHANNELS             8
#define FEATURE_CHANNELS            12
#define DOWNSAMPLE_CHANNELS         4
#define UPSAMPLE_CHANNELS           32

#define FILTER_SIZE_3X3             9
#define FILTER_SIZE1_3X3            3
#define FILTER_SIZE_1X1             1
#define FILTER_SIZE1_1X1            1

#define TRIPCOUNT_WIN_INPUT 		(INPUT_HEIGHT * INPUT_WIDTH * NR_OF_INPUT_CHANNELS + (INPUT_WIDTH + 1) * NR_OF_INPUT_CHANNELS)
#define TRIPCOUNT_WIN_FEATURE 		(INPUT_HEIGHT * INPUT_WIDTH * FEATURE_CHANNELS + (INPUT_WIDTH + 1) * FEATURE_CHANNELS)
#define TRIPCOUNT_WIN_MIDDLE 		(INPUT_HEIGHT * INPUT_WIDTH * MIDDLE_CHANNELS + (INPUT_WIDTH + 1) * MIDDLE_CHANNELS)

#define INPUT_RX_SIZE               (INPUT_HEIGHT * RX_LENGTH * NR_OF_INPUT_CHANNELS)
#define INPUT_GRID_SIZE             (INPUT_HEIGHT * INPUT_WIDTH * NR_OF_INPUT_CHANNELS)
#define FEATURE_SIZE                (INPUT_HEIGHT * INPUT_WIDTH * FEATURE_CHANNELS)
#define MIDDLE_SIZE					(INPUT_HEIGHT * INPUT_WIDTH * MIDDLE_CHANNELS)
#define DOWNSAMPLE_SIZE				(INPUT_HEIGHT * INPUT_WIDTH * DOWNSAMPLE_CHANNELS)

#define OUTPUT_SIZE                 (OUTPUT_HEIGHT * OUTPUT_WIDTH * NR_OF_INPUT_CHANNELS)

#define RANGE_SIGMOID               3
#define RANGE_SIGMOID_DIV           (NR_OF_TABLE_SIGMOID / (2 * RANGE_SIGMOID))
#define NR_OF_TABLE_SIGMOID         512

///---------------------------------------------------------------------------------------------------------------------
///Declare the data types
///---------------------------------------------------------------------------------------------------------------------
#define FLOAT_TOTAL_WIDTH           32
#define FLOAT_INT_WIDTH             7
typedef ap_fixed<FLOAT_TOTAL_WIDTH, FLOAT_INT_WIDTH> F32;
typedef hls::stream<F32> stream_data;
#define FLOAT_TOTAL_WIDTH_2       	16
#define FLOAT_INT_WIDTH_2          	10
typedef ap_fixed<FLOAT_TOTAL_WIDTH_2, FLOAT_INT_WIDTH_2> F16;
///---------------------------------------------------------------------------------------------------------------------
typedef unsigned char      		U8;
typedef unsigned short     		U16;
typedef unsigned int       		U32;

typedef signed char        		I8;
typedef signed short       		I16;
typedef signed int         		I32;
typedef hls::stream<U8> stream_u8;
typedef hls::stream<I8> stream_i8;
typedef hls::stream<I8> stream_weight;
///---------------------------------------------------------------------------------------------------------------------
struct window_u8 {
    U8 pix[FILTER_SIZE1_3X3][FILTER_SIZE1_3X3];
};
struct window_i8 {
	I8 pix[FILTER_SIZE1_3X3][FILTER_SIZE1_3X3];
};
typedef hls::stream<window_i8> stream_window;
///---------------------------------------------------------------------------------------------------------------------
typedef ap_axis<32,0,0,0> transPkt;
// For converting byte-wise from float to integer and back, because hls streams use ints
union fp_int {
	int i;
	float fp;
};
typedef hls::stream<transPkt> stream_pkt;


///---------------------------------------------------------------------------------------------------------------------
///Declare the top function
///---------------------------------------------------------------------------------------------------------------------
void SwiftChannel(stream_pkt &stream_in, stream_pkt &stream_out);

#endif