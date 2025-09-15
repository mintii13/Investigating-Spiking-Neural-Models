# Unlocking the Potential of Spike-based Transformer Architecture: Investigating Spiking Neural Models for Classification Tasks

Nguyen Minh Tri, Huynh Cong Viet Ngu  
FPT University, Ho Chi Minh Campus, Vietnam  
minhtringuyen130205@gmail.com, nguhcv@fe.edu.vn

---

## Abstract

Spiking Neural Networks (SNNs) have emerged as a promising alternative to traditional Artificial Neural Networks (ANNs), offering higher energy efficiency and better alignment with biological neural systems. Among recent advancements, the Spike-driven Transformer architecture combines the energy efficiency of SNNs with the powerful feature extraction capabilities of Transformers, achieving state-of-the-art performance in image classification tasks. However, the exclusive use of the Leaky Integrate-and-Fire (LIF) neuron model in this architecture introduces inefficiencies due to its inherent "leak" mechanism, which increases computational overhead and limits responsiveness. To address these limitations, this study investigates alternative spiking neuron models, specifically the Integrate-and-Fire (IF) variants: IF Hard Reset and IF Soft Reset. These models eliminate the leak mechanism, allowing the membrane potential to accumulate inputs without decay. Experimental results on CIFAR-10 and CIFAR-100 datasets demonstrate that the IF Soft Reset model consistently outperforms both the LIF and IF Hard Reset models across different classification complexities. This improvement highlights the advantage of the gradual reset mechanism, which retains residual excitation and enhances temporal processing capabilities. The findings underscore the importance of carefully designing spiking neuron models to balance biological realism and computational efficiency.

## Requirements

```python3
timm == 0.6.12
pytorch == 2.4.1+cu121
cupy == 12.3.0
spikingjelly == 0.0.0.0.12
tensorboard == 2.14.0
```

> !!! Please install the spikingjelly and tensorboard correctly before raising issues about requirements. !!!

## Comparison of Methods with Spike Neuron Models

| Spike Neuron Model | $u_{th}$ | $V_{reset}$ | Beta ($\beta$) | Accuracy (%) |  |
|--------------------|----------|-------------|----------------|--------------|--------------|
|                    |          |             |                | **CIFAR-10** | **CIFAR-100** |
| LIF                | 1        | 0           | 1/2            | 94.34        | 76.00        |
| IF hard reset      | 0.8      | 0           | -              | 94.13        | 76.11        |
|                    | 1        |             |                | 94.44        | 76.07        |
|                    | 1.2      |             |                | 94.32        | 76.15        |
| IF soft reset      | 0.8      | -           | -              | 93.88        | 76.00        |
|                    | 1        |             |                | **94.53**    | **76.36**    |
|                    | 1.2      |             |                | 94.40        | **76.56**    |

## Train & Test

The hyper-parameters are in `./conf/`.

### CIFAR-10 Training:
```shell
# LIF model
python train.py -c conf/cifar10/2_256_300E_t4.yml --model sdt --spike-mode lif

# IF Hard Reset model
python train.py -c conf/cifar10/2_256_300E_t4.yml --model sdt --spike-mode if

# IF Soft Reset model
python train.py -c conf/cifar10/2_256_300E_t4.yml --model sdt --spike-mode if_soft
```

### CIFAR-100 Training:
```shell
# LIF model
python train.py -c conf/cifar100/2_256_300E_t4.yml --model sdt --spike-mode lif

# IF Hard Reset model
python train.py -c conf/cifar100/2_256_300E_t4.yml --model sdt --spike-mode if

# IF Soft Reset model
python train.py -c conf/cifar100/2_256_300E_t4.yml --model sdt --spike-mode if_soft
```

### Testing:
```shell
python firing_num.py -c /the/path/of/conf --model sdt --spike-mode if_soft --resume /the/path/of/parameters --no-resume-opt
```

## Data Prepare

- Use `PyTorch` to load the CIFAR-10 and CIFAR-100 datasets automatically
- Tree structure in `./data/`:

```shell
.
├── cifar-10-batches-py
└── cifar-100-python
```

## Model Checkpoints

The trained model checkpoints and detailed experimental results reported in the paper are available at:
**[Google Drive - Model Checkpoints](https://drive.google.com/drive/folders/1uWTPokbWw6EMd80KUTwjU8wVAnarxP_T?usp=sharing)**

This repository contains:
- Pre-trained model weights for all three neuron types (LIF, IF Hard Reset, IF Soft Reset)
- Results for both CIFAR-10 and CIFAR-100 datasets
- Training logs and configuration files
- Performance analysis and comparison data

## Contact Information

For help or issues using this repository, please submit a GitHub issue.
For other communications related to this project, please contact `minhtringuyen130205@gmail.com`.
