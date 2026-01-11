# OCS Project Optimization Report

## Overview

This project implements an **optimized version** of the Online Coreset Selection (OCS) algorithm. By applying modern deep learning optimization techniques, we achieved significant performance improvements in training speed.

> **Test Environment**: Google Colab with Tesla T4 GPU, PyTorch 2.9.0+cu126

---

## Objectives

The original OCS algorithm requires computing **per-sample gradients** for sample selection, which creates a performance bottleneck in traditional implementations. This optimization work aims to:

1. Speed up per-sample gradient computation
2. Improve overall GPU utilization
3. Reduce memory overhead

---

## Optimization Techniques

### 1. Vectorized Gradient Computation (torch.func.vmap)

**Original Method (autograd_hacks):**

```python
# Uses hooks during backward pass to collect per-sample gradients
autograd_hacks.add_hooks(model)
loss.backward(retain_graph=True)
autograd_hacks.compute_grad1(model)
```

**Optimized Method (vmap + grad):**

```python
from torch.func import functional_call, vmap, grad

def compute_per_sample_grads_vectorized(model, criterion, data, target, task_id):
    params = {k: v.detach().requires_grad_(True) for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def compute_grad_for_sample(sample, target_single):
        return grad(
            lambda p: compute_loss_stateless(p, buffers, model, criterion, sample, target_single, task_id)
        )(params)

    # vmap vectorizes at compile level, enabling batch parallelism
    per_sample_grads_dict = vmap(compute_grad_for_sample, in_dims=(0, 0))(data, target)
    return flatten_grads(per_sample_grads_dict)
```

**How it works:**

- `vmap` automatically vectorizes single-sample operations into batch operations
- Compared to autograd_hacks' Python-level hooks, vmap achieves parallelism at the C++/CUDA level
- Avoids the computation graph retention overhead from `retain_graph=True`

---

### 2. Mixed Precision Training (AMP)

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

with autocast('cuda'):
    pred = model(data, task_id)
    loss = criterion(pred, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**

- FP16 computation is 2-8x faster than FP32 (depending on hardware)
- Memory usage reduced by approximately 50%
- Numerical stability maintained through GradScaler

---

### 3. PyTorch 2.0 Compilation (torch.compile)

```python
model = torch.compile(model, mode='reduce-overhead')
```

**Compilation modes:**

- `default`: Balanced compilation time and performance
- `reduce-overhead`: Reduces Python overhead, suitable for small batches
- `max-autotune`: Maximum performance, longer compilation time

---

### 4. Other Micro-optimizations

| Optimization     | Original                        | Optimized                               | Benefit            |
| ---------------- | ------------------------------- | --------------------------------------- | ------------------ |
| Gradient zeroing | `optimizer.zero_grad()`         | `optimizer.zero_grad(set_to_none=True)` | Reduced memory ops |
| Data transfer    | `.to(DEVICE)`                   | `.to(DEVICE, non_blocking=True)`        | Async transfer     |
| Deep copy        | `copy.deepcopy(tensor)`         | `tensor.clone()`                        | Reduced overhead   |
| Eval mode        | Explicit `with torch.no_grad()` | `@torch.no_grad()` decorator            | Cleaner code       |

---

## Benchmark Results

### Test Environment

- **GPU**: NVIDIA Tesla T4 (Google Colab)
- **PyTorch**: 2.9.0+cu126
- **CUDA**: 12.6
- **Dataset**: Rotated MNIST (3 tasks)

### End-to-End Training Performance

| Configuration  | Time (s) | Speedup   | Accuracy |
| -------------- | -------- | --------- | -------- |
| Original       | 324.62   | 1.00x     | 96.81%   |
| vmap           | 203.28   | **1.60x** | 97.00%   |
| vmap + AMP     | 201.47   | 1.61x     | 96.37%   |
| vmap + compile | 198.85   | 1.63x     | 95.96%   |
| Full Optimized | 198.60   | 1.63x     | 96.58%   |

---

## Key Findings

1. **vmap vectorization** shows the most significant improvement, achieving **1.60x speedup** while maintaining high accuracy
2. **AMP and torch.compile** provide marginal additional gains, pushing the total speedup to **1.63x**
3. Combined optimizations performed slightly better than pure vmap in this benchmark
4. All optimized versions maintain comparable accuracy (~96-97%), proving that optimizations don't affect convergence

---

## Prerequisites

```bash
pip install -r requirements.txt
```

**Required packages:**

- PyTorch >= 2.0 (for torch.func support)
- torchvision
- numpy
- scipy

---

## Run

```bash
# Run the optimized version (recommended)
python ocs_mnist_vectorized.py

# Run performance benchmark
python benchmark_optimization.py

# Run the original version (for comparison)
python ocs_mnist.py
```

---

## Project Structure

```
OCS/
├── core/
│   ├── train_methods_mnist.py           # Original version
│   ├── train_methods_mnist_vectorized.py # vmap optimized version
│   └── autograd_hacks.py                # Legacy per-sample gradients
├── ocs_mnist.py                         # Original entry point
├── ocs_mnist_vectorized.py              # vmap version entry point
└── benchmark_optimization.py            # Performance benchmark script
```

---

## Acknowledgments

- Original OCS implementation: [jaehong31/OCS](https://github.com/jaehong31/OCS)
- Built upon: [imirzadeh/stable-continual-learning](https://github.com/imirzadeh/stable-continual-learning)
- Bilevel coresets: [zalanborsos/bilevel_coresets](https://github.com/zalanborsos/bilevel_coresets)

For the original project README with paper details, see [README_original.md](README_original.md).

---

## References

- [PyTorch torch.func Documentation](https://pytorch.org/docs/stable/func.html)
- [Automatic Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

---

## Citation

This optimization work is based on the following paper:

```bibtex
@inproceedings{yoon2022online,
    title={Online Coreset Selection for Rehearsal-based Continual Learning},
    author={Jaehong Yoon and Divyam Madaan and Eunho Yang and Sung Ju Hwang},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/pdf?id=f9D-5WNG4Nv}
}
```

---

## Author

WU, Yu-Chen (Lima)

Date: 2026/01/11
