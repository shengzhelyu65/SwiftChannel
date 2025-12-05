# SwiftChannel

<p align="center"> <em><b>A Novel Algorithm-Hardware Co-design for 5G MIMO Channel Estimation</b></em> <p>
<p align="center"> <em><b>(5G MIMO System Simulation + Deep Learning Algorithm + FPGA-Based Hardware Implementation)</b></em> <p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Platform](https://img.shields.io/badge/platform-FPGA-orange)](#)
[![stars](https://img.shields.io/github/stars/shengzhelyu65/SwiftChannel.svg?style=flat-square)](https://github.com/shengzhelyu65/SwiftChannel/stargazers)
![GitHub repo size](https://img.shields.io/github/repo-size/shengzhelyu65/SwiftChannel.svg?style=flat-square)

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Simulation](#simulation)
  - [Software](#software)
  - [Hardware](#hardware)
- [License](#license)

## Requirements

### Software Dependencies

| Tool | Version | Purpose |
|------|---------|---------|
| [MATLAB](https://www.mathworks.com/products/matlab.html) | R2024b | System simulation and dataset generation |
| [Python](https://www.python.org/downloads/) | 3.x | Deep learning model training and testing |
| [Vitis](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html) | 2020.1+ (2024.1 recommended) | High-level synthesis and application host code |
| [Vivado](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-suite.html) | 2020.1+ (2024.1 recommended) | FPGA design and block design |

### Hardware Dependencies

The hardware implementation was tested on the AMD ZYNQ RFSoC XCZU49DR on the Avnet ADRS1000 board. But the HLS design can also be applied to other FPGA boards.

---

## Project Structure

```
SwiftChannel/
├── Simulation/          # MATLAB simulation scripts
├── Data/                # Training and testing datasets
├── Software/            # Python training and testing code
└── Hardware/            # FPGA implementation files
    ├── Vitis_HLS/       # High-level synthesis code
    ├── Vivado/          # Block design and constraints
    └── Vitis/           # Application host code
```

---

## Getting Started

### Simulation

The `Simulation/` folder contains MATLAB scripts for generating synthetic datasets under various SNR and UE velocity conditions.

**Files:**
- `training_dataset_generation_symbol_m.m` - Generate training datasets
- `testing_dataset_generation_symbol_m.m` - Generate testing datasets

**Usage:**
1. Open MATLAB and navigate to the `Simulation/` directory
2. Run the appropriate script for your dataset needs
3. Generated datasets will be saved in the `Data/` folder

---

### Software

The `Software/` folder contains the complete deep learning pipeline for training, compressing, and testing the SwiftChannel models and baseline models.

#### Directory Structure

```
Software/
├── config_file/         # Configuration files (YAML)
├── Dataloader/          # Data loading utilities
├── Models/              # Neural network model definitions
├── Baselines/           # Baseline model implementations
├── Metrics/             # Evaluation metrics
├── Outputs/             # Saved model weights for hardware
├── Experiments/         # Training logs and checkpoints
├── 1_train_dataset.py   # Generate training dataset
├── 1_test_dataset.py    # Generate testing dataset
├── 2_main.py           # Train teacher model
├── 2_knowledge_distillation.py  # Knowledge distillation
├── 3_convert_rp.py     # Re-parameterization
├── 3_qat.py            # Quantization-aware training
├── 4_test_before_qat.py # Test non-quantized model
├── 4_test_after_qat.py  # Test quantized model
├── 5_save_weights.py    # Export weights for hardware
├── 6_train_baseline.py  # Train baseline models
└── 6_test_baseline.py  # Test baseline models
```

#### Workflow

Follow these steps in order:

##### 1. Construct Datasets
```bash
# Construct training dataset
python 1_train_dataset.py

# Construct testing dataset
python 1_test_dataset.py
```
Constructed datasets are saved in the `Data/` folder.

##### 2. Train Teacher Model
```bash
python 2_main.py
```
- Training settings can be modified in `config_file/model_config.yaml` and `config_file/files_config.yaml`
- Trained models and logs are saved in `Experiments/`

##### 3. Knowledge Distillation
```bash
# Apply knowledge distillation to create student model
python 2_knowledge_distillation.py

# Apply re-parameterization
python 3_convert_rp.py
```

##### 4. Quantization-Aware Training
```bash
python 3_qat.py
```

##### 5. Test Models
```bash
# Test non-quantized model
python 4_test_before_qat.py

# Test quantized model (Final employed model)
python 4_test_after_qat.py
```

##### 6. Export Weights
```bash
python 5_save_weights.py
```
Exported weights are saved in `Outputs/` for hardware implementation.

#### Baseline Models

Train and test baseline models for comparison:

```bash
# Train a baseline model (e.g., ChannelNet, I_ResNet, Channelformer)
python 6_train_baseline.py <model_name>

# Test a baseline model
python 6_test_baseline.py
```

Model-specific hyperparameters can be configured in `Baselines/baseline_config.yaml`.

---

### Hardware

The `Hardware/` folder contains the HLS implementation and corresponding block design and application host code for the SwiftChannel model.

#### Directory Structure

```
Hardware/
├── Vitis_HLS/    # HLS implementation (C++/C)
│                 # - LS estimator
│                 # - SwiftChannel algorithm
├── Vivado/       # Block design and Arm PS presets
└── Vitis/        # Application host code for testing
```

#### Usage

1. **Vitis_HLS**: Contains High-Level Synthesis (HLS) code for both the Least Squares (LS) estimator and SwiftChannel algorithm. Synthesize the C++/C code into hardware IP using Vitis HLS.

2. **Vivado**: Contains the block design connecting HLS IP with other Zynq RFSoC components. Includes preset TCL files for the Arm Processing System (PS).

3. **Vitis**: Contains the Vitis application host code for testing SwiftChannel on the hardware platform.

---

## License

Copyright © 2025 [Shengzhe Lyu](https://github.com/shengzhelyu65).

This project is licensed under the [MIT License](LICENSE.txt).
