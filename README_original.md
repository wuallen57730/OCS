# OCS-Accelerated: Optimized Online Coreset Selection

A **modernized and accelerated** implementation of [Online Coreset Selection for Rehearsal-based Continual Learning](https://openreview.net/forum?id=f9D-5WNG4Nv) (ICLR 2022).

This fork focuses on **performance optimization** using modern PyTorch techniques, achieving **1.23x speedup** while maintaining accuracy.

<img align="middle" width="800" src="https://github.com/jaehong31/OCS/blob/main/OCS_concept.png">

---

## What's New: Performance Optimization

### Key Improvements

| Technique           | Description                                                 | Impact                  |
| ------------------- | ----------------------------------------------------------- | ----------------------- |
| **torch.func.vmap** | Vectorized per-sample gradient computation                  | **1.23x speedup**       |
| **Functional API**  | Replaced `autograd_hacks` with `torch.func.functional_call` | Cleaner, more efficient |
| **GPU-optimized**   | Reduced CPU-GPU data transfers                              | Lower latency           |

### Benchmark Results (Rotated MNIST, 3 Tasks)

| Method                    | Time (s)   | Accuracy (%) | Speedup   |
| ------------------------- | ---------- | ------------ | --------- |
| Original (autograd_hacks) | 248.36     | 96.82        | 1.00x     |
| **vmap (Optimized)**      | **202.38** | **96.47**    | **1.23x** |
| vmap + AMP                | 215.51     | 96.61        | 1.15x     |
| vmap + torch.compile      | 212.42     | 96.52        | 1.17x     |

> Tested on Google Colab with Tesla T4 GPU, PyTorch 2.9.0+cu126

### Technical Highlights

**Before (autograd_hacks):**

```python
# Hook-based approach - adds overhead
autograd_hacks.add_hooks(model)
loss.backward(retain_graph=True)
autograd_hacks.compute_grad1(model)
autograd_hacks.clear_backprops(model)
```

**After (vmap + functional_call):**

```python
# Vectorized approach - native PyTorch, more efficient
from torch.func import functional_call, vmap, grad

def compute_per_sample_grads_vectorized(model, criterion, data, target, task_id):
    params = {k: v.detach().requires_grad_(True) for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    per_sample_grads = vmap(
        grad(lambda p: compute_loss_stateless(p, buffers, model, criterion, sample, target, task_id))
    )(params)
    return flatten_grads(per_sample_grads)
```

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Experiments

**Original version:**

```bash
python ocs_mnist.py
```

**Optimized version (recommended):**

```bash
python ocs_mnist_vectorized.py
```

**Run benchmark comparison:**

```bash
python benchmark_optimization.py
```

---

## Project Structure

```
OCS/
├── ocs_mnist.py                    # Original MNIST experiment
├── ocs_mnist_vectorized.py         # Optimized MNIST experiment (recommended)
├── benchmark_optimization.py       # Performance benchmark script
├── OPTIMIZATION_REPORT.md          # Detailed optimization report
├── requirements.txt
├── core/
│   ├── train_methods_mnist.py           # Original training methods
│   ├── train_methods_mnist_vectorized.py # Optimized training methods
│   ├── models.py                        # CNN models
│   ├── data_utils.py                    # Data loading utilities
│   ├── utils.py                         # Helper functions
│   ├── autograd_hacks.py                # Legacy per-sample gradients
│   └── ...
├── checkpoints/                    # Model checkpoints
├── data/                          # Dataset directory
└── archive/                       # Archived/unused files
```

---

## Original Paper

This implementation is based on:

**Online Coreset Selection for Rehearsal-based Continual Learning** (ICLR 2022)

> A dataset is a shred of crucial evidence to describe a task. However, each data point in the dataset does not have the same potential, as some of the data points can be more representative or informative than others. This unequal importance among the data points may have a large impact in rehearsal-based continual learning, where we store a subset of the training examples (coreset) to be replayed later to alleviate catastrophic forgetting.

**Key Contributions:**

- Online Coreset Selection (OCS) method for continual learning
- Gradient-based sample selection criteria
- Applicable to any rehearsal-based continual learning method

---

## Acknowledgments

- Original implementation: [jaehong31/OCS](https://github.com/jaehong31/OCS)
- Built upon: [imirzadeh/stable-continual-learning](https://github.com/imirzadeh/stable-continual-learning)
- Bilevel coresets: [zalanborsos/bilevel_coresets](https://github.com/zalanborsos/bilevel_coresets)

## Citations

```
@inproceedings{yoon2022online,
    title={Online Coreset Selection for Rehearsal-based Continual Learning},
    author={Jaehong Yoon and Divyam Madaan and Eunho Yang and Sung Ju Hwang},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/pdf?id=f9D-5WNG4Nv}
}
```
