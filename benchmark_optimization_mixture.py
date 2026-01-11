"""
Mixture Dataset Optimization Benchmark Test
Compares performance improvements from different optimization technique combinations

Test Items:
1. Original (compute_and_flatten_example_grads) - Original version
2. vmap only - Vectorization only
3. vmap + AMP - Vectorization + Mixed Precision
4. vmap + compile - Vectorization + Compilation
5. Full Optimized - All optimizations combined
"""
import os
import sys
import torch
import time
import argparse
import numpy as np
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models import ResNet18
from core.utils import DEVICE, save_model, save_task_model_by_policy


def print_header():
    print("\n" + "=" * 70)
    print("OCS Mixture Dataset Optimization Benchmark")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70 + "\n")


def cleanup():
    """Clean up GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_base_config(args):
    """Base configuration for Mixture dataset"""
    return {
        'num_tasks': args.tasks,
        'n_tasks': args.tasks,
        'n_classes': [0, 10, 10, 10, 43, 10],  # Mixture dataset class distribution
        'stream_size': args.stream_size,
        'batch_size': args.batch_size,
        'memory_size': args.memory_size,
        'dataset': 'mixture',
        'mlp_hiddens': 256,
        'dropout': 0.1,
        'seq_epochs': 1,
        'seq_lr': 0.1,
        'lr_decay': 0.85,
        'momentum': 0.8,
        'ocspick': True,
        'select_type': 'ocs_select',
        'tau': 1000.0,
        'is_r2c': True,
        'r2c_iter': 100,
        'coreset_base': False,
        'ref_hyp': 0.1,
    }


def get_mixture_loaders(config):
    """Get loaders for Mixture dataset"""
    from core.data_utils import MixtureDataset as MD
    is_imb = True if 'imb' in config['dataset'] else False
    md = MD(config, is_imbalanced=is_imb)
    return md.get_loader()


def benchmark_original(config, loaders, exp_dir):
    """Test original version"""
    print("\n" + "-" * 50)
    print("Testing: Original (compute_and_flatten_example_grads)")
    print("-" * 50)
    
    from core import train_methods_mixture as train_methods
    
    os.makedirs(exp_dir, exist_ok=True)
    config['exp_dir'] = exp_dir
    init_model = ResNet18(nclasses=83, config=config)
    save_model(init_model, f'{exp_dir}/init.pth')
    
    cleanup()
    start_time = time.time()
    
    for task in range(1, config['n_tasks'] + 1):
        model = train_methods.train_task_sequentially(task, loaders, config)
        save_task_model_by_policy(model, task, 'seq', exp_dir)
        print(f"  Task {task} completed")
    
    total_time = time.time() - start_time
    
    # Evaluation
    metrics = train_methods.eval_single_epoch(model, loaders['sequential'][config['n_tasks']]['val'], config)
    
    cleanup()
    return total_time, metrics['accuracy']


def benchmark_vmap_only(config, loaders, exp_dir):
    """Test vmap optimization only"""
    print("\n" + "-" * 50)
    print("Testing: vmap only")
    print("-" * 50)
    
    from core import train_methods_mixture_optimized as train_methods
    train_methods.USE_COMPILE = False
    train_methods.USE_AMP = False
    train_methods.reset_compiled_functions()
    
    os.makedirs(exp_dir, exist_ok=True)
    config['exp_dir'] = exp_dir
    init_model = ResNet18(nclasses=83, config=config)
    save_model(init_model, f'{exp_dir}/init.pth')
    
    cleanup()
    start_time = time.time()
    
    for task in range(1, config['n_tasks'] + 1):
        model = train_methods.train_task_sequentially(task, loaders, config)
        save_task_model_by_policy(model, task, 'seq', exp_dir)
        print(f"  Task {task} completed")
    
    total_time = time.time() - start_time
    
    metrics = train_methods.eval_single_epoch(model, loaders['sequential'][config['n_tasks']]['val'], config)
    
    cleanup()
    return total_time, metrics['accuracy']


def benchmark_vmap_amp(config, loaders, exp_dir):
    """Test vmap + AMP"""
    print("\n" + "-" * 50)
    print("Testing: vmap + AMP")
    print("-" * 50)
    
    from core import train_methods_mixture_optimized as train_methods
    train_methods.USE_COMPILE = False
    train_methods.USE_AMP = True
    train_methods.reset_compiled_functions()
    
    os.makedirs(exp_dir, exist_ok=True)
    config['exp_dir'] = exp_dir
    init_model = ResNet18(nclasses=83, config=config)
    save_model(init_model, f'{exp_dir}/init.pth')
    
    cleanup()
    start_time = time.time()
    
    for task in range(1, config['n_tasks'] + 1):
        model = train_methods.train_task_sequentially(task, loaders, config)
        save_task_model_by_policy(model, task, 'seq', exp_dir)
        print(f"  Task {task} completed")
    
    total_time = time.time() - start_time
    
    metrics = train_methods.eval_single_epoch(model, loaders['sequential'][config['n_tasks']]['val'], config)
    
    cleanup()
    return total_time, metrics['accuracy']


def benchmark_vmap_compile(config, loaders, exp_dir):
    """Test vmap + compile"""
    print("\n" + "-" * 50)
    print("Testing: vmap + torch.compile")
    print("-" * 50)
    
    from core import train_methods_mixture_optimized as train_methods
    train_methods.USE_COMPILE = True
    train_methods.USE_AMP = False
    train_methods.reset_compiled_functions()
    
    os.makedirs(exp_dir, exist_ok=True)
    config['exp_dir'] = exp_dir
    init_model = ResNet18(nclasses=83, config=config)
    save_model(init_model, f'{exp_dir}/init.pth')
    
    cleanup()
    start_time = time.time()
    
    for task in range(1, config['n_tasks'] + 1):
        model = train_methods.train_task_sequentially(task, loaders, config)
        save_task_model_by_policy(model, task, 'seq', exp_dir)
        print(f"  Task {task} completed")
    
    total_time = time.time() - start_time
    
    metrics = train_methods.eval_single_epoch(model, loaders['sequential'][config['n_tasks']]['val'], config)
    
    cleanup()
    return total_time, metrics['accuracy']


def benchmark_full_optimized(config, loaders, exp_dir):
    """Test all optimizations combined"""
    print("\n" + "-" * 50)
    print("Testing: Full Optimized (vmap + AMP + compile)")
    print("-" * 50)
    
    from core import train_methods_mixture_optimized as train_methods
    train_methods.USE_COMPILE = True
    train_methods.USE_AMP = True
    train_methods.reset_compiled_functions()
    
    os.makedirs(exp_dir, exist_ok=True)
    config['exp_dir'] = exp_dir
    init_model = ResNet18(nclasses=83, config=config)
    save_model(init_model, f'{exp_dir}/init.pth')
    
    cleanup()
    start_time = time.time()
    
    for task in range(1, config['n_tasks'] + 1):
        model = train_methods.train_task_sequentially(task, loaders, config)
        save_task_model_by_policy(model, task, 'seq', exp_dir)
        print(f"  Task {task} completed")
    
    total_time = time.time() - start_time
    
    metrics = train_methods.eval_single_epoch(model, loaders['sequential'][config['n_tasks']]['val'], config)
    
    cleanup()
    return total_time, metrics['accuracy']


def benchmark_per_sample_grads(stream_size=50):
    """
    Standalone test for per-sample gradients computation speedup
    This test best demonstrates the vmap advantage for Mixture (ResNet)
    """
    print("\n" + "=" * 70)
    print("Per-Sample Gradients Performance Comparison (Mixture/ResNet)")
    print(f"   (stream_size = {stream_size})")
    print("=" * 70)
    
    import torch.nn as nn
    from core.models import ResNet18
    from core.utils import compute_and_flatten_example_grads
    
    # Mixture uses ResNet with 83 classes total
    config = {'n_classes': [0, 10, 10, 10, 43, 10], 'n_tasks': 5, 'mlp_hiddens': 256, 'dropout': 0.1}
    model = ResNet18(nclasses=83, config=config).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    # Generate test data (32x32 RGB images like CIFAR)
    data = torch.randn(stream_size, 3, 32, 32).to(DEVICE)
    target = torch.randint(0, 10, (stream_size,)).to(DEVICE)  # Task 1 has 10 classes
    task_id = 1
    
    print("\nWarming up...")
    
    # Test original method
    print("\nOriginal method (compute_and_flatten_example_grads):")
    
    model_orig = ResNet18(nclasses=83, config=config).to(DEVICE)
    
    times_orig = []
    for _ in range(5):
        model_orig.eval()
        model_orig.zero_grad()
        
        start = time.time()
        eg_orig = compute_and_flatten_example_grads(model_orig, criterion, data, target, task_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_orig.append(time.time() - start)
    
    avg_orig = np.mean(times_orig[1:])  # Skip first run (warmup)
    print(f"   Avg time: {avg_orig*1000:.2f} ms")
    print(f"   Gradient shape: {eg_orig.shape}")
    
    # Test vectorized method
    print("\nVectorized method (vmap):")
    from core.train_methods_mixture_optimized import compute_per_sample_grads_vectorized
    
    model_vmap = ResNet18(nclasses=83, config=config).to(DEVICE)
    
    times_vmap = []
    for _ in range(5):
        start = time.time()
        eg_vmap = compute_per_sample_grads_vectorized(model_vmap, criterion, data, target, task_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_vmap.append(time.time() - start)
    
    avg_vmap = np.mean(times_vmap[1:])
    print(f"   Avg time: {avg_vmap*1000:.2f} ms")
    print(f"   Gradient shape: {eg_vmap.shape}")
    
    # Comparison
    speedup = avg_orig / avg_vmap
    print(f"\nSpeedup: {speedup:.2f}x")
    
    return avg_orig, avg_vmap, speedup


def main():
    parser = argparse.ArgumentParser(description='Mixture Dataset Optimization Benchmark Test')
    parser.add_argument('--tasks', type=int, default=3, help='Number of tasks (max 5 for mixture)')
    parser.add_argument('--stream_size', type=int, default=20, help='Stream size')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--memory_size', type=int, default=83, help='Coreset memory size')
    parser.add_argument('--quick', action='store_true', help='Only test per-sample gradients')
    parser.add_argument('--large_batch', action='store_true', help='Use larger stream_size (100)')
    args = parser.parse_args()
    
    # Ensure tasks don't exceed maximum for mixture
    args.tasks = min(args.tasks, 5)
    
    print_header()
    
    # Quick test - best demonstrates optimization effect
    if args.quick or args.large_batch:
        stream_size = 100 if args.large_batch else args.stream_size
        benchmark_per_sample_grads(stream_size)
        return
    
    # Full end-to-end test
    config = get_base_config(args)
    
    print("Loading Mixture dataset...")
    loaders = get_mixture_loaders(config)
    
    results = {}
    
    # 1. Original
    results['Original'] = benchmark_original(
        config.copy(), loaders, 
        'checkpoints/mixture_bench_original'
    )
    
    # 2. vmap only
    results['vmap'] = benchmark_vmap_only(
        config.copy(), loaders,
        'checkpoints/mixture_bench_vmap'
    )
    
    # 3. vmap + AMP
    results['vmap+AMP'] = benchmark_vmap_amp(
        config.copy(), loaders,
        'checkpoints/mixture_bench_vmap_amp'
    )
    
    # 4. vmap + compile
    if hasattr(torch, 'compile'):
        results['vmap+compile'] = benchmark_vmap_compile(
            config.copy(), loaders,
            'checkpoints/mixture_bench_vmap_compile'
        )
    
    # 5. Full optimized
    if hasattr(torch, 'compile'):
        results['Full Optimized'] = benchmark_full_optimized(
            config.copy(), loaders,
            'checkpoints/mixture_bench_full'
        )
    
    # Results summary
    print("\n" + "=" * 70)
    print("Mixture Dataset Results Summary")
    print("=" * 70)
    print(f"{'Method':<20} {'Time (s)':<15} {'Accuracy (%)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    baseline_time = results['Original'][0]
    for method, (total_time, accuracy) in results.items():
        speedup = baseline_time / total_time
        print(f"{method:<20} {total_time:<15.2f} {accuracy:<15.2f} {speedup:<10.2f}x")
    
    print("=" * 70)
    
    # Standalone per-sample gradients test
    print("\n" + "=" * 70)
    print("Per-Sample Gradients Standalone Test (Best demonstrates vmap advantage)")
    print("=" * 70)
    benchmark_per_sample_grads(args.stream_size)


if __name__ == '__main__':
    main()
