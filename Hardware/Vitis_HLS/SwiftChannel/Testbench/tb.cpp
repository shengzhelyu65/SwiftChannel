#include "wrapper.hpp"
#include <fstream>
#include <iostream>

#define TB_RX_LENGTH 16
#define TB_TX_LENGTH 2
#define TB_HEIGHT 108
#define TB_WIDTH 32
#define TB_WAVEFORM 512

vector<float> readBinaryFile(const string& file_path) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Could not open binary file.");
    }

    file.seekg(0, ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);

    vector<float> buffer(file_size / sizeof(float));
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    cout << "Size of the buffer: " << buffer.size() << " floats" << endl;
    
    return buffer;
}

void bufferToStream(const vector<float>& buffer, hls::stream<transPkt>& stream, int height, int width, int channel) {
    int pixel_count = 0;
    fp_int tmp_data;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channel; ++c) {
                int index = (y * width + x) * channel + c;
                float value = buffer[index];

                tmp_data.fp = value;

                transPkt pkg;
                pkg.data = tmp_data.i;
                pkg.last = (x == width - 1) && (c == channel - 1) ? 1 : 0;
                stream.write(pkg);

                pixel_count++;
            }
        }
    }

    cout << "Number of data written to stream: " << pixel_count << endl;
}

void streamToBuffer(hls::stream<transPkt>& stream, vector<float>& buffer, int height, int width, int channel) {
    int pixel_count = 0;
    fp_int tmp_data;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channel; ++c) {
                transPkt pkg;
                stream.read(pkg);
                tmp_data.i = pkg.data;

                float value = tmp_data.fp;

                buffer.push_back(value);

                pixel_count++;
            }
        }
    }

    cout << "Number of pixels written to buffer: " << pixel_count << endl;
}

void writeBinaryFile(const string& file_path, const vector<float>& buffer) {
    ofstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Could not open binary file for writing.");
    }

    // Write the entire buffer to the file
    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(float));
    file.close();
    cout << "Written " << buffer.size() << " floats to binary file." << endl;
}

int main(int argc, char** argv) {
    /// *** Test the SwiftChannel function
    hls::stream<transPkt> src_hw, sink_hw;

    string file_name;
    string binary_file_path;
    vector<float> buffer;

    cout << "Reading binary file..." << endl;

    // Test whole
    int scale_factor = 4;
    binary_file_path = "rx_grid.bin";
    buffer = readBinaryFile(binary_file_path);
    bufferToStream(buffer, src_hw, TB_HEIGHT, TB_RX_LENGTH, 2);
    buffer.clear();
    cout << "Hardware processing..." << endl;

    SwiftChannel(src_hw, sink_hw);

    cout << "Finish hardware processing..." << endl;
    file_name = "h_estimated_output.bin";
    streamToBuffer(sink_hw, buffer, TB_HEIGHT*scale_factor, TB_WIDTH*scale_factor, 2);

    cout << "Writing binary file..." << endl;
    writeBinaryFile(file_name, buffer);

    cout << "Done!" << endl;

    vector<float> ground_truth;
    float threshold = 0.1;

    binary_file_path = "h_output.bin";
    ground_truth = readBinaryFile(binary_file_path);

    bool is_equal = true;
    int error_count = 0;
    for (int i = 0; i < buffer.size(); i++) {
        if ((buffer[i] - ground_truth[i]) > threshold || (ground_truth[i] - buffer[i]) > threshold) {
            error_count++;
        }
    }
    if (error_count > 10) {
        is_equal = false;
        cout << "Error count: " << error_count << endl;
        return 1;
    }

    return 0;
}
