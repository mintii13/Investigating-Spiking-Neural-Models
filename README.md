# Unlocking the Potential of Spike-based Transformer Architecture: Investigating Spiking Neural Models for Classification Task

Nguyen Minh Tri, Huynh Cong Viet Ngu  
FPT University, Ho Chi Minh Campus, Vietnam  
minhtringuyen130205@gmail.com, nguhcv@fe.edu.vn
---

## Abstract

Spiking Neural Networks (SNNs) have emerged as a promising alternative to traditional Artificial Neural
Networks (ANNs), offering higher energy efficiency and better alignment with biological neural systems. Among
recent advancements, the Spike-driven Transformer architecture combines the energy efficiency of SNNs with the
powerful feature extraction capabilities of Transformers, achieving state-of-the-art performance in image classifi-
cation tasks. However, the exclusive use of the Leaky Integrate-and-Fire (LIF) neuron model in this architecture
introduces inefficiencies due to its inherent ”leak” mechanism, which increases computational overhead and limits
responsiveness. To address these limitations, this study investigates alternative spiking neuron models, specifically
the Integrate-and-Fire (IF) variants: IF Hard Reset and IF Soft Reset. These models eliminate the leak mechanism,
allowing the membrane potential to accumulate inputs without decay. Experimental results demonstrate that the IF
Soft Reset model outperforms both the LIF and IF Hard Reset models, achieving a classification accuracy of 94.38%
compared to 94.34% for the others. This improvement highlights the advantage of the gradual reset mechanism, which
retains residual excitation and enhances temporal processing capabilities. The findings underscore the importance of
carefully designing spiking neuron models to balance biological realism and computational efficiency. 

## Requirements

```python3
timm == 0.6.12
1.10.0 <= pytorch < 2.0.0
cupy
spikingjelly == 0.0.0.0.12
tensorboard
```

> !!! Please install the spikingjelly and tensorboard correctly before raising issues about requirements. !!!

## Comparison of Methods with Spike Neuron Models

| Methods                  | Spike Neuron Model | Batch size | Timestep | $u_{th}$ | $V_{reset}$  | Beta ($\beta$)| Accuracy |
|--------------------------|--------------------|------------|----------|----------|--------------|---------------|----------|
| Spike-driven Transformer | LIF                | 64         | 4        | 1        | 0            | 1/2           | 94.34    |
| Spike-driven Transformer | IF hard reset      | 64         | 4        | 1        | 0            | -             | 94.34    |
| Spike-driven Transformer | IF soft reset      | 64         | 4        | 1        | -            | -             | **94.38**    |

## Train & Test

The hyper-parameters are in `./conf/`.


Train:

```shell
python train.py -c conf/cifar10/2_256_300E_t4.yml --model sdt --spike-mode if_soft
```

Test:

```shell
python firing_num.py -c /the/path/of/conf --model sdt --spike-mode if_soft --resume /the/path/of/parameters --no-resume-opt

```


## Data Prepare

- use `PyTorch` to load the CIFAR10 dataset

Tree in `./data/`.

```shell
.
├── cifar-10-batches-py

```



## Contact Information


For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact `minhtringuyen130205@gmail.com`.
