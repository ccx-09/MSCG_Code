#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Union
from tqdm import tqdm
import time
logger = logging.getLogger(__name__)

class TensorPatchDataset(Dataset):

    def __init__(self, images: List[np.ndarray], patch_size: int=64, stride: int=32):
        self.patch_size = patch_size
        self.stride = stride
        self.patch_info = []
        for img_idx, image in enumerate(images):
            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.images_tensor = getattr(self, 'images_tensor', [])
            self.images_tensor.append(torch.from_numpy(image_rgb.transpose(2, 0, 1)))
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    self.patch_info.append((img_idx, y, x, h, w))

    def __len__(self):
        return len(self.patch_info)

    def __getitem__(self, idx):
        img_idx, y, x, h, w = self.patch_info[idx]
        image_tensor = self.images_tensor[img_idx]
        patch = image_tensor[:, y:y + self.patch_size, x:x + self.patch_size]
        patch_resized = F.interpolate(patch.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        patch_normalized = normalize(patch_resized)
        coord = torch.tensor([y + self.patch_size // 2, x + self.patch_size // 2], dtype=torch.float32)
        return (patch_normalized, coord, torch.tensor(img_idx, dtype=torch.long))

class GPUOptimizedCNNExtractor:

    def __init__(self, model_type: str='vgg16', device: str='auto', batch_size: int=64, num_workers: int=4, mixed_precision: bool=True, use_tf32: bool=True):
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        if use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        if self.mixed_precision and self.device.type == 'cuda':
            pass
        if self.device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1000000000.0

    def _load_model(self) -> nn.Module:
        if self.model_type == 'vgg16':
            model = models.vgg16(weights='IMAGENET1K_V1')
            self.feature_dim = 4096
            feature_model = nn.Sequential(model.features, model.avgpool, nn.Flatten(), model.classifier[:2])
        elif self.model_type == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
            self.feature_dim = 2048
            feature_model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        elif self.model_type == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V1')
            self.feature_dim = 2048
            feature_model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        else:
            raise ValueError(f'Unsupported model: {self.model_type}')
        sample_input = torch.randn(1, 3, 224, 224)
        optimized_model = torch.jit.trace(feature_model, sample_input)
        return optimized_model

    def extract_features_gpu_optimized(self, images: List[np.ndarray], patch_size: int=64, stride: int=32) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if not images:
            return ([], [])
        start_time = time.time()
        dataset = TensorPatchDataset(images, patch_size, stride)
        dl_kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=False)
        if self.num_workers and self.num_workers > 0:
            dl_kwargs.update(dict(prefetch_factor=2, persistent_workers=False))
        dataloader = DataLoader(dataset, **dl_kwargs)
        all_patch_features = []
        all_patch_coords = []
        all_patch_img_ids = []
        with torch.no_grad():
            for batch_patches, batch_coords, batch_img_ids in tqdm(dataloader, desc='GPU Processing'):
                batch_patches = batch_patches.to(self.device, non_blocking=True)
                if self.mixed_precision and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        batch_features = self.model(batch_patches)
                else:
                    batch_features = self.model(batch_patches)
                all_patch_features.append(batch_features.cpu())
                all_patch_coords.append(batch_coords)
                all_patch_img_ids.append(batch_img_ids)
        all_features_tensor = torch.cat(all_patch_features, dim=0).numpy()
        all_coords_tensor = torch.cat(all_patch_coords, dim=0).numpy()
        all_img_ids_tensor = torch.cat(all_patch_img_ids, dim=0).numpy()
        features_by_image = []
        coords_by_image = []
        for img_idx in range(len(images)):
            mask = all_img_ids_tensor == img_idx
            img_features = all_features_tensor[mask]
            img_coords = all_coords_tensor[mask]
            features_by_image.append(img_features)
            coords_by_image.append(img_coords)
        elapsed_time = time.time() - start_time
        total_patches = len(all_features_tensor)
        speed = total_patches / elapsed_time if elapsed_time > 0 else 0
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return (features_by_image, coords_by_image)

    def extract_single_image(self, image: np.ndarray, patch_size: int=64, stride: int=32) -> Tuple[np.ndarray, np.ndarray]:
        features_list, coords_list = self.extract_features_gpu_optimized([image], patch_size, stride)
        return (features_list[0], coords_list[0])

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def benchmark(self, image_size: Tuple[int, int]=(512, 512), patch_size: int=64, stride: int=32, num_images: int=5) -> dict:
        sample_images = []
        for _ in range(num_images):
            image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            sample_images.append(image)
        _ = self.extract_features_gpu_optimized(sample_images[:1], patch_size, stride)
        start_time = time.time()
        features_list, coords_list = self.extract_features_gpu_optimized(sample_images, patch_size, stride)
        elapsed_time = time.time() - start_time
        total_patches = sum((len(f) for f in features_list))
        patches_per_sec = total_patches / elapsed_time
        images_per_sec = num_images / elapsed_time
        memory_used = memory_cached = 0
        if self.device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1000000000.0
            memory_cached = torch.cuda.max_memory_reserved() / 1000000000.0
        metrics = {'total_images': num_images, 'total_patches': total_patches, 'elapsed_time': elapsed_time, 'patches_per_second': patches_per_sec, 'images_per_second': images_per_sec, 'memory_used_gb': memory_used, 'memory_cached_gb': memory_cached, 'batch_size': self.batch_size, 'num_workers': self.num_workers, 'mixed_precision': self.mixed_precision, 'optimization_level': 'gpu_native'}
        return metrics

def benchmark_comparison():
    image_size = (512, 512)
    patch_size = 64
    stride = 32
    num_images = 3
    try:
        extractor_new = GPUOptimizedCNNExtractor(model_type='vgg16', batch_size=64, num_workers=4, mixed_precision=True)
        metrics_new = extractor_new.benchmark(image_size, patch_size, stride, num_images)
        from optimized_cnn_extractor import OptimizedCNNFeatureExtractor
        extractor_old = OptimizedCNNFeatureExtractor(model_type='vgg16', batch_size=64, mixed_precision=True)
        metrics_old = extractor_old.benchmark_performance(image_size, patch_size, stride, num_images)
        speedup = metrics_new['patches_per_second'] / metrics_old['patches_per_second']
        memory_improvement = (metrics_old['memory_used_gb'] - metrics_new['memory_used_gb']) / metrics_old['memory_used_gb'] * 100
    except Exception as e:
        extractor = GPUOptimizedCNNExtractor()
        metrics = extractor.benchmark(image_size, patch_size, stride, num_images)
