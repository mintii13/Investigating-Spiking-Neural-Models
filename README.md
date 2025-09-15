# Unlocking the Potential of Spike-based Transformer Architecture: Investigating Spiking Neural Models for Classification Tasks

Nguyen Minh Tri, Huynh Vuong Khang, Huynh Cong Viet Ngu  
FPT University, Ho Chi Minh Campus, Vietnam  
minhtringuyen130205@gmail.com, khanghvse184160@fpt.edu.vn, nguhcv@fe.edu.vn

---

## Abstract

Spiking Neural Networks (SNNs) offer energy-efficient and biologically plausible alternatives to conventional Artificial Neural Networks (ANNs). The Spike-driven Transformer architecture integrates SNN efficiency with the Transformer-based feature extraction, achieving competitive results for image classification. However, its reliance on the Leaky Integrate-and-Fire (LIF) neuron introduces computational overhead due to the leak mechanism. This work investigates alternative neuron models—**IF Hard Reset** and **IF Soft Reset**—which remove the leak dynamics to improve efficiency. We conduct systematic evaluations on CIFAR-10 and CIFAR-100 datasets, analyzing accuracy, inference speed, spike activity patterns and energy consumption across different neuron models. Experimental results show that **IF Soft Reset** achieves the highest accuracy at 94.53%, 76.56% compared to 94.44%, 76.15% of Hard Reset and 94.34%, 76.00% of LIF on CIFAR-10, CIFAR-100 respectively. It also has fastest inference speed as 1323.4 FPS and 12.09 ms latency, outperforming IF Hard Reset with 1244.2 FPS and LIF with 1161.2 FPS. The improvement is attributed to its gradual reset behavior, which preserves residual excitation and enhances temporal processing. These findings offer practical design guidelines for deploying efficient spike-based Transformers under resource-constrained environments.

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
