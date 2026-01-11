"""
Vectorized MNIST Training Methods
Uses torch.func's vmap + grad for efficient per-sample gradient computation
Compared to the original autograd_hacks, this achieves better GPU utilization
"""
import pdb
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributions as td
from torch.func import functional_call, vmap, grad

from .utils import DEVICE, save_model, load_model
from .utils import flatten_params, flatten_grads, assign_weights, assign_grads, accum_grads
from .mode_connectivity import get_line_loss, get_coreset_loss, reconstruct_coreset_loader2
from .summary import Summarizer

import copy

gumbel_dist = td.gumbel.Gumbel(0, 1)

coreset_methods = ['uniform', 'coreset',
           'kmeans_features', 'kcenter_features', 'kmeans_grads',
           'kmeans_embedding', 'kcenter_embedding', 'kcenter_grads',
           'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching']

# Lazy import for NTK components
_kernel_fn = None
_bc = None

def _init_ntk_components():
    """Lazy initialization of NTK-related components"""
    global _kernel_fn, _bc
    if _kernel_fn is None:
        from .ntk_generator import generate_cnn_ntk
        from .bilevel_coreset import BilevelCoreset
        
        def coreset_cross_entropy(K, alpha, y, weights, lmbda):
            loss = torch.nn.CrossEntropyLoss(reduction='none')
            loss_value = torch.mean(loss(torch.matmul(K, alpha), y.long()) * weights)
            if lmbda > 0:
                loss_value += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
            return loss_value
        
        _kernel_fn = lambda x, y: generate_cnn_ntk(x.reshape(-1, 28, 28, 1), y.reshape(-1, 28, 28, 1))
        _bc = BilevelCoreset(outer_loss_fn=coreset_cross_entropy,
                inner_loss_fn=coreset_cross_entropy, out_dim=10, max_outer_it=1,
                max_inner_it=200, logging_period=10)
    return _kernel_fn, _bc


# ============================================================================
# Vectorized Per-Sample Gradients Computation (Core Optimization)
# ============================================================================

def compute_loss_stateless(params, buffers, model, criterion, sample, target, task_id):
    """
    Stateless loss computation function (single sample version)
    """
    batch_sample = sample.unsqueeze(0)
    batch_target = target.unsqueeze(0)
    predictions = functional_call(model, (params, buffers), (batch_sample, task_id))
    loss = criterion(predictions, batch_target.long())
    return loss


def compute_per_sample_grads_vectorized(model, criterion, data, target, task_id):
    """
    Vectorized computation of per-sample gradients using vmap + grad
    
    Compared to autograd_hacks:
    - autograd_hacks: Uses hooks during backward to collect gradients
    - vmap: Vectorizes at compile level, more efficient GPU utilization
    """
    # Important: Must set to eval mode, otherwise dropout causes errors in vmap
    was_training = model.training
    model.eval()
    
    params = {k: v.detach().requires_grad_(True) for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    
    def compute_grad_for_sample(sample, target_single):
        return grad(
            lambda p: compute_loss_stateless(p, buffers, model, criterion, sample, target_single, task_id)
        )(params)
    
    per_sample_grads_dict = vmap(compute_grad_for_sample, in_dims=(0, 0))(data, target)
    
    # Flatten gradients
    grads_list = []
    for name, param in model.named_parameters():
        if 'bn' not in name and 'IC' not in name:
            g = per_sample_grads_dict[name]
            grads_list.append(g.view(g.size(0), -1))
    
    # Restore original training mode
    if was_training:
        model.train()
    
    return torch.cat(grads_list, dim=1)


# ============================================================================
# Sample Selection Algorithm
# ============================================================================

def sample_selection(g, eg, config, ref_grads=None, attn=None):
    """OCS sample selection"""
    ng = torch.norm(g)
    neg = torch.norm(eg, dim=1)
    mean_sim = torch.matmul(g, eg.t()) / torch.maximum(ng * neg, torch.ones_like(neg) * 1e-6)
    negd = torch.unsqueeze(neg, 1)

    cross_div = torch.matmul(eg, eg.t()) / torch.maximum(torch.matmul(negd, negd.t()), torch.ones_like(negd) * 1e-6)
    avg_div = torch.mean(cross_div, 0)

    coreset_aff = 0.
    if ref_grads is not None:
        ref_ng = torch.norm(ref_grads)
        coreset_aff = torch.matmul(ref_grads, eg.t()) / torch.maximum(ref_ng * neg, torch.ones_like(neg) * 1e-6)

    measure = mean_sim - avg_div + config['tau'] * coreset_aff
    _, u_idx = torch.sort(measure, descending=True)
    return u_idx.cpu().numpy()


def classwise_fair_selection(task, cand_target, sorted_index, num_per_label, config, is_shuffle=True):
    """Class-balanced sample selection"""
    num_examples_per_task = config['memory_size'] // task
    num_examples_per_class = num_examples_per_task // config['n_classes']
    num_residuals = num_examples_per_task - num_examples_per_class * config['n_classes']
    residuals = np.sum([(num_examples_per_class - n_c) * (num_examples_per_class > n_c) for n_c in num_per_label])
    num_residuals += residuals

    while True:
        n_less_sample_class = np.sum([(num_examples_per_class > n_c) for n_c in num_per_label])
        num_class = (config['n_classes'] - n_less_sample_class)
        if (num_residuals // num_class) > 0:
            num_examples_per_class += (num_residuals // num_class)
            num_residuals -= (num_residuals // num_class) * num_class
        else:
            break

    selected = []
    for j in range(config['n_classes']):
        position = np.squeeze((cand_target[sorted_index] == j).nonzero())
        if position.numel() > 1:
            selected.append(position[:num_examples_per_class])
        elif position.numel() == 0:
            continue
        else:
            selected.append([position])

    selected = np.concatenate(selected)
    unselected = np.array(list(set(np.arange(num_examples_per_task)) ^ set(selected)))
    final_num_residuals = num_examples_per_task - len(selected)
    best_residuals = unselected[:final_num_residuals]
    selected = np.concatenate([selected, best_residuals])

    if is_shuffle:
        random.shuffle(selected)

    return sorted_index[selected.astype(int)]


# ============================================================================
# Coreset Selection and Update
# ============================================================================

def select_coreset(loader, task, model, candidates, config, candidate_size=1000, fair_selection=True):
    """Select coreset samples"""
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    if fair_selection:
        cand_data, cand_target = [], []
        cand_size = len(candidates)
        for batch_idx, (data, target, task_id) in enumerate(loader['sequential'][task]['train']):
            if batch_idx == cand_size:
                break
            cand_data.append(data[candidates[batch_idx]])
            cand_target.append(target[candidates[batch_idx]])

        cand_data = torch.cat(cand_data, 0)
        cand_target = torch.cat(cand_target, 0)

        random_pick_up = torch.randperm(len(cand_target))[:candidate_size]
        cand_data = cand_data[random_pick_up]
        cand_target = cand_target[random_pick_up]

        num_per_label = [len((cand_target == jj).nonzero()) for jj in range(config['n_classes'])]
        num_examples_per_task = config['memory_size'] // task

        if config['select_type'] in coreset_methods:
            rs = np.random.RandomState(0)
            if config['select_type'] == 'coreset':
                kernel_fn, bc = _init_ntk_components()
                pick, _ = bc.build_with_representer_proxy_batch(
                    cand_data.cpu().numpy(), cand_target.cpu().numpy(),
                    num_examples_per_task, kernel_fn, cache_kernel=True, start_size=1, inner_reg=1e-3)
            else:
                summarizer = Summarizer.factory(config['select_type'], rs)
                pick = summarizer.build_summary(cand_data.cpu().numpy(), cand_target.cpu().numpy(),
                                                num_examples_per_task, method=config['select_type'],
                                                model=model, device=DEVICE)
            loader['coreset'][task]['train'].data = copy.deepcopy(cand_data[pick])
            loader['coreset'][task]['train'].targets = copy.deepcopy(cand_target[pick])
        else:
            # Use vectorized method
            _eg = compute_per_sample_grads_vectorized(model, criterion, cand_data.to(DEVICE), cand_target.to(DEVICE), task)
            _g = torch.mean(_eg, 0)
            sorted_idx = sample_selection(_g, _eg, config)
            
            if config['select_type'] == 'ocs_select':
                pick = torch.randperm(len(sorted_idx))
                selected = classwise_fair_selection(task, cand_target, pick, num_per_label, config, is_shuffle=True)

            loader['coreset'][task]['train'].data = copy.deepcopy(cand_data[selected])
            loader['coreset'][task]['train'].targets = copy.deepcopy(cand_target[selected])


def update_coreset(loader, task, model, task_id, config):
    """Update coreset samples"""
    num_examples_per_task = config['memory_size'] // task
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    for tid in range(1, task):
        if config['select_type'] in coreset_methods:
            tid_coreset = loader['coreset'][tid]['train'].data
            tid_targets = loader['coreset'][tid]['train'].targets
            num_per_label = [len((tid_targets.cpu() == jj).nonzero()) for jj in range(config['n_classes'])]
            rs = np.random.RandomState(0)
            if config['select_type'] == 'coreset':
                kernel_fn, bc = _init_ntk_components()
                selected, _ = bc.build_with_representer_proxy_batch(
                    loader['coreset'][tid]['train'].data.cpu().numpy(),
                    loader['coreset'][tid]['train'].targets.cpu().numpy(),
                    num_examples_per_task, kernel_fn, cache_kernel=True, start_size=1, inner_reg=1e-3)
            else:
                summarizer = Summarizer.factory(config['select_type'], rs)
                selected = summarizer.build_summary(
                    loader['coreset'][tid]['train'].data.cpu().numpy(),
                    loader['coreset'][tid]['train'].targets.cpu().numpy(),
                    num_examples_per_task, method=config['select_type'], model=model, device=DEVICE)
        elif config['select_type'] == 'ocs_select':
            tid_coreset = loader['coreset'][tid]['train'].data
            tid_targets = loader['coreset'][tid]['train'].targets

            # Use vectorized method
            _tid_eg = compute_per_sample_grads_vectorized(model, criterion, tid_coreset.to(DEVICE), tid_targets.to(DEVICE), task_id)
            _tid_g = torch.mean(_tid_eg, 0)
            sorted_idx = sample_selection(_tid_g, _tid_eg, config)

            num_per_label = [len((tid_targets.cpu() == jj).nonzero()) for jj in range(config['n_classes'])]
            selected = classwise_fair_selection(task, tid_targets, sorted_idx, num_per_label, config)

        loader['coreset'][tid]['train'].data = copy.deepcopy(loader['coreset'][tid]['train'].data[selected])
        loader['coreset'][tid]['train'].targets = copy.deepcopy(loader['coreset'][tid]['train'].targets[selected])


# ============================================================================
# Training Steps (Vectorized Version)
# ============================================================================

def train_single_step(model, optimizer, loader, task, step, config):
    """Single task training step"""
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    is_last_step = True if step == config['n_substeps'] else False
    rs = np.random.RandomState(0)
    if config['select_type'] in coreset_methods and config['select_type'] != 'coreset':
        summarizer = Summarizer.factory(config['select_type'], rs)

    candidates_indices = []
    for batch_idx, (data, target, task_id) in enumerate(loader['sequential'][task]['train']):
        model.train()
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        is_rand_start = True if ((step == 1) and (batch_idx < config['r2c_iter']) and config['is_r2c']) else False
        is_ocspick = True if (config['ocspick'] and len(data) > config['batch_size']) else False

        if is_ocspick and not is_rand_start:
            # Vectorized per-sample gradients computation
            model.eval()
            _eg = compute_per_sample_grads_vectorized(model, criterion, data, target, task_id)
            _g = torch.mean(_eg, 0)
            sorted_idx = sample_selection(_g, _eg, config)

            pick = sorted_idx[:config['batch_size']]
            model.train()
            optimizer.zero_grad()

            pred = model(data[pick], task_id)
            loss = criterion(pred, target[pick])
            loss.backward()

            if is_last_step:
                candidates_indices.append(pick)

        elif config['select_type'] in coreset_methods:
            size = min(len(data), config['batch_size'])
            if config['select_type'] == 'coreset':
                if is_last_step and batch_idx < 10:
                    kernel_fn, bc = _init_ntk_components()
                    selected_pick, _ = bc.build_with_representer_proxy_batch(
                        data.cpu().numpy(), target.cpu().numpy(),
                        config['batch_size'], kernel_fn, cache_kernel=True, start_size=1, inner_reg=1e-3)
                else:
                    selected_pick = torch.randperm(len(data))[:size]
            else:
                selected_pick = summarizer.build_summary(data.cpu().numpy(), target.cpu().numpy(),
                                                         config['batch_size'], method=config['select_type'],
                                                         model=model, device=DEVICE)
            pick = selected_pick
            pred = model(data[pick], task_id)
            loss = criterion(pred, target[pick])
            loss.backward()
            if is_last_step:
                candidates_indices.append(pick)
        else:
            size = min(len(data), config['batch_size'])
            pick = torch.randperm(len(data))[:size]
            pred = model(data[pick], task_id)
            loss = criterion(pred, target[pick])
            loss.backward()

        optimizer.step()

    if is_last_step:
        select_coreset(loader, task, model, candidates_indices, config)

    return model


def train_ocs_single_step(model, optimizer, loader, task, step, config):
    """OCS training step (vectorized version)"""
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    is_last_step = True if step == config['n_substeps'] else False
    prev_coreset = [loader['coreset'][tid]['train'].data for tid in range(1, task)]
    prev_targets = [loader['coreset'][tid]['train'].targets for tid in range(1, task)]
    c_x = torch.cat(prev_coreset, 0)
    c_y = torch.cat(prev_targets, 0)

    ref_loader = reconstruct_coreset_loader2(config, loader['coreset'], task - 1)
    ref_iterloader = iter(ref_loader)

    candidates_indices = []
    for batch_idx, (data, target, task_id) in enumerate(loader['sequential'][task]['train']):
        model.train()
        optimizer.zero_grad()
        is_rand_start = True if ((step == 1) and (batch_idx < config['r2c_iter']) and config['is_r2c']) else False

        # Compute reference gradients
        ref_pred = model(c_x.to(DEVICE), None)
        ref_loss = criterion(ref_pred, c_y.long().to(DEVICE))
        ref_loss.backward()
        ref_grads = copy.deepcopy(flatten_grads(model))
        optimizer.zero_grad()

        data = data.to(DEVICE)
        target = target.to(DEVICE)

        if is_rand_start:
            size = min(len(data), config['batch_size'])
            pick = torch.randperm(len(data))[:size]
        else:
            # Vectorized per-sample gradients computation
            model.eval()
            _eg = compute_per_sample_grads_vectorized(model, criterion, data, target, task_id)
            _g = torch.mean(_eg, 0)
            sorted_idx = sample_selection(_g, _eg, config, ref_grads=ref_grads)
            pick = sorted_idx[:config['batch_size']]
            model.train()
            optimizer.zero_grad()

        pred = model(data[pick], task_id)
        loss = criterion(pred, target[pick])

        try:
            ref_data = next(ref_iterloader)
        except StopIteration:
            ref_iterloader = iter(ref_loader)
            ref_data = next(ref_iterloader)

        ref_loss = get_coreset_loss(model, ref_data, config)
        loss += config['ref_hyp'] * ref_loss
        loss.backward()
        optimizer.step()

        if is_last_step:
            candidates_indices.append(pick)

    if is_last_step:
        select_coreset(loader, task, model, candidates_indices, config)
        update_coreset(loader, task, model, task_id, config)

    return model


def train_coreset_single_step(model, optimizer, loader, task, step, config):
    """Coreset baseline training"""
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    is_last_step = True if step == config['n_substeps'] else False
    prev_coreset = [loader['coreset'][tid]['train'].data for tid in range(1, task)]
    prev_targets = [loader['coreset'][tid]['train'].targets for tid in range(1, task)]
    c_x = torch.cat(prev_coreset, 0)
    c_y = torch.cat(prev_targets, 0)

    ref_loader = reconstruct_coreset_loader2(config, loader['coreset'], task - 1)
    ref_iterloader = iter(ref_loader)
    rs = np.random.RandomState(0)
    if config['select_type'] in coreset_methods and config['select_type'] != 'coreset':
        summarizer = Summarizer.factory(config['select_type'], rs)

    candidates_indices = []
    for batch_idx, (data, target, task_id) in enumerate(loader['sequential'][task]['train']):
        model.train()
        optimizer.zero_grad()

        data = data.to(DEVICE)
        target = target.to(DEVICE)
        size = min(config['batch_size'], len(data))
        pick = summarizer.build_summary(data.cpu().numpy(), target.cpu().numpy(), size,
                                        method=config['select_type'], model=model, device=DEVICE)
        pred = model(data[pick], task_id)
        loss = criterion(pred, target[pick])

        try:
            ref_data = next(ref_iterloader)
        except StopIteration:
            ref_iterloader = iter(ref_loader)
            ref_data = next(ref_iterloader)

        ref_loss = get_coreset_loss(model, ref_data, config)
        loss += config['ref_hyp'] * ref_loss
        loss.backward()

        if is_last_step:
            if config['select_type'] == 'coreset':
                if batch_idx < 10:
                    kernel_fn, bc = _init_ntk_components()
                    selected_pick, _ = bc.build_with_representer_proxy_batch(
                        data.cpu().numpy(), target.cpu().numpy(),
                        config['batch_size'], kernel_fn, cache_kernel=True, start_size=1, inner_reg=1e-3)
            candidates_indices.append(pick)
        optimizer.step()

    if is_last_step:
        select_coreset(loader, task, model, candidates_indices, config)
        update_coreset(loader, task, model, task_id, config)
    return model


# ============================================================================
# Evaluation and Main Training Flow
# ============================================================================

def eval_single_epoch(net, loader, config):
    """Evaluation"""
    net = net.to(DEVICE)
    net.eval()
    test_loss = 0
    correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    with torch.no_grad():
        for data, target, task_id in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            count += len(target)
            output = net(data, task_id)

            test_loss += criterion(output, target).item() * len(target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            correct_bool = pred.eq(target.data.view_as(pred))
            for cid in range(10):
                cid_index = torch.where(target == cid, torch.ones_like(target), torch.zeros_like(target))
                class_correct[cid] += (cid_index.data.view_as(correct_bool) * correct_bool).sum().item()
                class_total[cid] += cid_index.sum().item()

    test_loss /= count
    correct = correct.to('cpu')
    avg_acc = 100.0 * float(correct.numpy()) / count

    pc_avg_acc = [np.round(a / (b + 1e-10), 4) for a, b in zip(class_correct, class_total)]
    return {'accuracy': avg_acc, 'per_class_accuracy': pc_avg_acc, 'loss': test_loss}


def train_task_sequentially(task, train_loader, config, summary=None):
    """Sequential training (vectorized version - no autograd_hacks needed)"""
    EXP_DIR = config['exp_dir']

    current_lr = config['seq_lr'] * (config['lr_decay']) ** (task - 1)
    prev_model_name = 'init' if task == 1 else 't_{}_seq'.format(str(task - 1))
    prev_model_path = '{}/{}.pth'.format(EXP_DIR, prev_model_name)
    model = load_model(prev_model_path).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=config['momentum'])

    # Note: Vectorized version does not require autograd_hacks.add_hooks(model)

    config['n_substeps'] = int(config['seq_epochs'] * (config['stream_size'] / config['batch_size']))
    for _step in range(1, config['n_substeps'] + 1):
        if config['coreset_base'] and task > 1:
            model = train_coreset_single_step(model, optimizer, train_loader, task, _step, config)
        elif task == 1 or (config['ocspick'] == False):
            model = train_single_step(model, optimizer, train_loader, task, _step, config)
        else:
            model = train_ocs_single_step(model, optimizer, train_loader, task, _step, config)
        metrics = eval_single_epoch(model, train_loader['sequential'][task]['val'], config)
        print('Epoch {} >> (per-task accuracy): {}'.format(_step / config['n_substeps'], np.mean(metrics['accuracy'])))
        print('Epoch {} >> (class accuracy): {}'.format(_step / config['n_substeps'], metrics['per_class_accuracy']))
    return model

