#!/usr/bin/env python3
import logging
from typing import Dict, Tuple, Optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

def get_gpu_memory_info() -> Dict[str, float]:
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {'total': 0, 'allocated': 0, 'free': 0}
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    allocated_memory = torch.cuda.memory_allocated(0) / 1024 ** 3
    free_memory = total_memory - allocated_memory
    return {'total': total_memory, 'allocated': allocated_memory, 'free': free_memory}

def get_memory_info() -> Dict[str, float]:
    return get_gpu_memory_info()

def estimate_optimal_batch_size(img_size: int=1024, model_name: str='vit_h') -> int:
    memory_info = get_gpu_memory_info()
    total_gb = memory_info['total']
    if model_name == 'vit_h':
        base_model_memory = 2.6
        per_sample_memory = 1.2
    elif model_name == 'vit_b':
        base_model_memory = 0.4
        per_sample_memory = 0.3
    else:
        base_model_memory = 3.0
        per_sample_memory = 1.5
    available_memory = max(total_gb - base_model_memory - 2.0, 1.0)
    optimal_batch_size = max(int(available_memory / per_sample_memory), 1)
    if total_gb >= 40:
        optimal_batch_size = min(optimal_batch_size, 16)
    elif total_gb >= 20:
        optimal_batch_size = min(optimal_batch_size, 8)
    else:
        optimal_batch_size = min(optimal_batch_size, 4)
    return optimal_batch_size

def configure_mixed_precision() -> Tuple[bool, str]:
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return (False, 'cpu')
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    major, minor = capability
    if major >= 8:
        dtype = 'bfloat16'
        enabled = True
    elif major >= 7:
        dtype = 'float16'
        enabled = True
    else:
        dtype = 'float32'
        enabled = False
    return (enabled, dtype)

def optimize_dataloader_settings(batch_size: int, num_samples: int) -> Dict[str, int]:
    memory_info = get_gpu_memory_info()
    if memory_info['total'] >= 40:
        num_workers = min(8, torch.get_num_threads() if TORCH_AVAILABLE else 8)
        prefetch_factor = 4
    elif memory_info['total'] >= 20:
        num_workers = min(4, torch.get_num_threads() if TORCH_AVAILABLE else 4)
        prefetch_factor = 3
    else:
        num_workers = 2
        prefetch_factor = 2
    if num_samples < 100:
        num_workers = min(num_workers, 2)
    if batch_size == 1:
        num_workers = min(num_workers, 2)
        prefetch_factor = 2
    return {'num_workers': num_workers, 'prefetch_factor': prefetch_factor, 'pin_memory': True, 'persistent_workers': num_workers > 0}

def clear_gpu_memory():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def optimize_memory_usage():
    clear_gpu_memory()
