# SwiftChannel

<p align="center"> <em><b>A Novel Algorithm-Hardware Co-design for 5G MIMO Channel Estimation</b></em> <p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Platform](https://img.shields.io/badge/platform-FPGA-orange)](#)

## Requirements
- [MATLAB R2024b](https://www.mathworks.com/products/matlab.html)
- [Python 3](https://www.python.org/downloads/)
- [Vitis Unified IDE (or Vitis HLS) + Vivado + Vitis IDE 2020.1 or later (recommended: 2024.1 for faster compilation)](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html)

## File Structure
```bash
SwiftChannel
├── Simulation
├── Data
├── Software
└── Hardware
```

### Simulation
```bash
Simulation
├── training_dataset_generation_symbol_m.m
└── testing_dataset_generation_symbol.m
```
This folder contains the simulation code to be run in MATLAB. The code is designed to generate synthetic datasets for the SwiftChannel algorithm training and testing under various SNR and UE velocity settings.

### Software
```bash
Software
├── config_file
├── Dataloader
├── Models
├── Metrics
├── Outputs
├── Experiments
├── 1_train_dataset.py
├── 1_test_dataset.py
├── 2_main.py
├── 2_knowledge_distillation.py
├── 3_convert_rp.py
├── 3_qat.py
├── 4_test_before_qat.py
├── 4_test_after_qat.py
└── 5_save_weights.py
```
This folder contains the code to generate the dataset, train the teacher model, and apply model compression to obtain the student model, which is the final deployment model. The code for testing the final model is also included in this folder. The code has been tested with the following software versions:
```python
python 3.12.7
pytorch 2.4.1
cuda 12.4
```

**Usage:**
1. **Generate the dataset**: Run `1_train_dataset.py` and `1_test_dataset.py` to generate the training and testing datasets, respectively. The generated datasets will be saved back in the `Data` folder.
2. **Train the teacher model**: Run `2_main.py` to train the teacher model. The trained model and training logs will be saved in the `Experiments` folder. The training settings can be modified by editing the `files_config.yaml` and `model_config.yaml` files in the `config_file` folder.
3. **Apply knowledge distillation**: Run `2_knowledge_distillation.py` to obtain the student model. And then run `3_convert_rp.py` to apply re-parameterization to the student model.
4. **Quantization-aware training**: Run `3_qat.py` to apply quantization-aware training to the student model.
5. **Test the model**: Run `4_test_before_qat.py` and `4_test_after_qat.py` to test the teacher model and student model, respectively.
6. **Save the model weights**: Run `5_save_weights.py` to save the model weights in a format that can be used for hardware implementation. The output will be saved in the `Outputs` folder.

## Hardware
```bash
Hardware
├── Vitis_HLS
├── Vivado
└── Vitis
```
The hardware implementation targets the AMD ZYNQ RFSoC XCZU49DR on the Avnet ADRS1000 board. The development environment is based on Vitis Unified IDE 2024.1, Vivado 2024.1, and Vitis Classic 2024.1.

**Usage:**
1. **Vitis_HLS**: This folder contains the HLS implementation of both the least squres (LS) estimator and SwiftChannel algorithm. The HLS code is written in C++/C and can be synthesized into hardware IP using Vitis HLS.
2. **Vivado**: This folder contains the block design created using Vivado to connect the HLS IP with other components in the Zynq RFSoC. The preset tcl file for the Arm PS is also included in this folder.
3. **Vitis**: This folder contains the Vitis application host code to test SwiftChannel.

## License

Copyright © 2025 [Shengzhe Lyu](https://github.com/shengzhelyu65).<br />
This project is [MIT](https://github.com/shengzhelyu65/SwiftChannel/LICENSE) licensed.
