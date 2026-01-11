"""
Optimization Benchmark Test
Compares performance improvements from different optimization technique combinations

Tested on: Google Colab with Tesla T4 GPU

Test Items:
1. Original (autograd_hacks) - Original version
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

from core.data_utils import get_all_loaders
from core.models import MLP
from core.utils import DEVICE, save_model, save_task_model_by_policy

def print_header():
    print("\n" + "=" * 70)
    print("OCS Optimization Benchmark (Google Colab - Tesla T4)")
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
    """Base configuration"""
    return {
        'num_tasks': args.tasks,
        'n_tasks': args.tasks,
        'n_classes': 10,
        'per_task_rotation': 9,
        'stream_size': args.stream_size,
        'batch_size': args.batch_size,
        'memory_size': args.memory_size,
        'dataset': 'rot-mnist',
        'mlp_hiddens': 256,
        'dropout': 0.2,
        'seq_epochs': 1,
        'seq_lr': 0.005,
        'lr_decay': 0.75,
        'momentum': 0.8,
        'ocspick': True,
        'select_type': 'ocs_select',
        'tau': 1000.0,
        'is_r2c': True,
        'r2c_iter': 100,
        'coreset_base': False,
        'ref_hyp': 10.0,
    }


def benchmark_original(config, loaders, exp_dir):
    """Test original version (autograd_hacks)"""
    print("\n" + "-" * 50)
    print("Testing: Original (autograd_hacks)")
    print("-" * 50)
    
    from core import train_methods_mnist as train_methods
    
    os.makedirs(exp_dir, exist_ok=True)
    config['exp_dir'] = exp_dir
    init_model = MLP(config)
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
    
    from core import train_methods_mnist_optimized as train_methods
    train_methods.USE_COMPILE = False
    train_methods.USE_AMP = False
    
    os.makedirs(exp_dir, exist_ok=True)
    config['exp_dir'] = exp_dir
    init_model = MLP(config)
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
    
    from core import train_methods_mnist_optimized as train_methods
    train_methods.USE_COMPILE = False
    train_methods.USE_AMP = True
    
    os.makedirs(exp_dir, exist_ok=True)
    config['exp_dir'] = exp_dir
    init_model = MLP(config)
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
    
    from core import train_methods_mnist_optimized as train_methods
    train_methods.USE_COMPILE = True
    train_methods.USE_AMP = False
    
    os.makedirs(exp_dir, exist_ok=True)
    config['exp_dir'] = exp_dir
    init_model = MLP(config)
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
    
    from core import train_methods_mnist_optimized as train_methods
    train_methods.USE_COMPILE = True
    train_methods.USE_AMP = True
    
    os.makedirs(exp_dir, exist_ok=True)
    config['exp_dir'] = exp_dir
    init_model = MLP(config)
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


def benchmark_per_sample_grads(stream_size=500):
    """
    Standalone test for per-sample gradients computation speedup
    This test best demonstrates the vmap advantage
    """
    print("\n" + "=" * 70)
    print("Per-Sample Gradients Performance Comparison")
    print(f"   (stream_size = {stream_size})")
    print("=" * 70)
    
    import torch.nn as nn
    from core.models import MLP
    
    config = {'n_classes': 10, 'n_tasks': 20, 'mlp_hiddens': 256, 'dropout': 0.2}
    model = MLP(config).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    # Generate test data
    data = torch.randn(stream_size, 784).to(DEVICE)
    target = torch.randint(0, 10, (stream_size,)).to(DEVICE)
    task_id = 1
    
    # Warm up
    print("\nWarming up...")
    
    # Test original method
    print("\nOriginal method (autograd_hacks):")
    from core import autograd_hacks
    
    model_orig = MLP(config).to(DEVICE)
    autograd_hacks.add_hooks(model_orig)
    
    times_orig = []
    for _ in range(5):
        model_orig.eval()
        model_orig.zero_grad()
        autograd_hacks.clear_backprops(model_orig)
        
        start = time.time()
        pred = model_orig(data, task_id)
        loss = criterion(pred, target)
        loss.backward(retain_graph=True)
        autograd_hacks.compute_grad1(model_orig)
        
        grads_list = []
        for name, param in model_orig.named_parameters():
            if 'bn' not in name and 'IC' not in name:
                grads_list.append(param.grad1.view(param.grad1.size(0), -1))
        eg_orig = torch.cat(grads_list, dim=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_orig.append(time.time() - start)
    
    # Note: remove_hooks has a bug, not calling it doesn't affect the test
    avg_orig = np.mean(times_orig[1:])  # Skip first run
    print(f"   Avg time: {avg_orig*1000:.2f} ms")
    
    # Test vectorized method
    print("\nVectorized method (vmap):")
    from core.train_methods_mnist_optimized import compute_per_sample_grads_vectorized
    
    model_vmap = MLP(config).to(DEVICE)
    
    times_vmap = []
    for _ in range(5):
        start = time.time()
        eg_vmap = compute_per_sample_grads_vectorized(model_vmap, criterion, data, target, task_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_vmap.append(time.time() - start)
    
    avg_vmap = np.mean(times_vmap[1:])
    print(f"   Avg time: {avg_vmap*1000:.2f} ms")
    
    # Comparison
    speedup = avg_orig / avg_vmap
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Verify correctness
    diff = torch.abs(eg_orig - eg_vmap).max().item()
    print(f"Max error: {diff:.2e} (numerical correctness verification)")
    
    return avg_orig, avg_vmap, speedup


def main():
    parser = argparse.ArgumentParser(description='Optimization Benchmark Test')
    parser.add_argument('--tasks', type=int, default=3, help='Number of tasks')
    parser.add_argument('--stream_size', type=int, default=100, help='Stream size')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--memory_size', type=int, default=300, help='Coreset memory size')
    parser.add_argument('--quick', action='store_true', help='Only test per-sample gradients')
    parser.add_argument('--large_batch', action='store_true', help='Use larger stream_size (500)')
    args = parser.parse_args()
    
    print_header()
    
    # Quick test - best demonstrates optimization effect
    if args.quick or args.large_batch:
        stream_size = 500 if args.large_batch else args.stream_size
        benchmark_per_sample_grads(stream_size)
        return
    
    # Full end-to-end test
    config = get_base_config(args)
    
    print("Loading dataset...")
    loaders = get_all_loaders(
        config['dataset'], config['num_tasks'],
        config['batch_size'], config['stream_size'],
        config['memory_size'], config.get('per_task_rotation')
    )
    
    results = {}
    
    # 1. Original
    results['Original'] = benchmark_original(
        config.copy(), loaders, 
        'checkpoints/bench_original'
    )
    
    # 2. vmap only
    results['vmap'] = benchmark_vmap_only(
        config.copy(), loaders,
        'checkpoints/bench_vmap'
    )
    
    # 3. vmap + AMP
    results['vmap+AMP'] = benchmark_vmap_amp(
        config.copy(), loaders,
        'checkpoints/bench_vmap_amp'
    )
    
    # 4. vmap + compile
    if hasattr(torch, 'compile'):
        results['vmap+compile'] = benchmark_vmap_compile(
            config.copy(), loaders,
            'checkpoints/bench_vmap_compile'
        )
    
    # 5. Full optimized
    if hasattr(torch, 'compile'):
        results['Full Optimized'] = benchmark_full_optimized(
            config.copy(), loaders,
            'checkpoints/bench_full'
        )
    
    # Results summary
    print("\n" + "=" * 70)
    print("Results Summary")
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
