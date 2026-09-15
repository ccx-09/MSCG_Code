#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Union
from tqdm import tqdm
import time
logger = logging.getLogger(__name__)

class OptimizedCNNFeatureExtractor:

    def __init__(self, model_type: str='vgg16', device: str='auto', batch_size: int=64, mixed_precision: bool=True, use_tf32: bool=True, memory_efficient: bool=True):
        self.model_type = model_type
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.memory_efficient = memory_efficient
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        if use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model = self._load_optimized_model()
        self.model.eval()
        self.model.to(self.device)
        if self.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        self.transform = self._create_optimized_transforms()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1000000000.0

    def _load_optimized_model(self) -> nn.Module:
        if self.model_type == 'vgg16':
            model = models.vgg16(weights='IMAGENET1K_V1')
            self.feature_dim = 4096
            optimized_model = nn.Sequential(model.features, model.avgpool, nn.Flatten(), model.classifier[:2])
        elif self.model_type == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
            self.feature_dim = 2048
            optimized_model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        elif self.model_type == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V1')
            self.feature_dim = 2048
            optimized_model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        else:
            raise ValueError(f'Unsupported model: {self.model_type}')
        if self.memory_efficient:
            sample_input = torch.randn(1, 3, 224, 224)
            optimized_model = torch.jit.trace(optimized_model, sample_input)
        return optimized_model

    def _create_optimized_transforms(self) -> transforms.Compose:
        return transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def extract_patches_batch(self, images: List[np.ndarray], patch_size: int=64, stride: int=32) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        all_features = []
        all_coords = []
        start_time = time.time()
        total_patches = 0
        for image_idx, image in enumerate(tqdm(images, desc='Processing images')):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            patches = []
            coords = []
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = image_rgb[y:y + patch_size, x:x + patch_size]
                    patches.append(patch)
                    coords.append([y + patch_size // 2, x + patch_size // 2])
            if len(patches) == 0:
                all_features.append(np.empty((0, self.feature_dim)))
                all_coords.append(np.empty((0, 2)))
                continue
            features = self._process_patches_batched(patches)
            all_features.append(features)
            all_coords.append(np.array(coords))
            total_patches += len(patches)
        elapsed_time = time.time() - start_time
        if total_patches > 0:
            speed = total_patches / elapsed_time
        return (all_features, all_coords)

    def _process_patches_batched(self, patches: List[np.ndarray]) -> np.ndarray:
        features_list = []
        for i in range(0, len(patches), self.batch_size):
            batch_patches = patches[i:i + self.batch_size]
            batch_tensors = []
            for patch in batch_patches:
                tensor = self.transform(patch)
                batch_tensors.append(tensor)
            batch_tensor = torch.stack(batch_tensors).to(self.device, non_blocking=True)
            with torch.no_grad():
                if self.mixed_precision and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        batch_features = self.model(batch_tensor)
                else:
                    batch_features = self.model(batch_tensor)
                features_numpy = batch_features.cpu().numpy()
                features_list.append(features_numpy)
            if i % (self.batch_size * 4) == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        return np.vstack(features_list)

    def extract_single_image_optimized(self, image: np.ndarray, patch_size: int=64, stride: int=32) -> Tuple[np.ndarray, np.ndarray]:
        features_list, coords_list = self.extract_patches_batch([image], patch_size, stride)
        return (features_list[0], coords_list[0])

    def benchmark_performance(self, image_size: Tuple[int, int]=(512, 512), patch_size: int=64, stride: int=32, num_images: int=5) -> dict:
        sample_images = []
        for _ in range(num_images):
            image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            sample_images.append(image)
        start_time = time.time()
        features_list, coords_list = self.extract_patches_batch(sample_images, patch_size, stride)
        elapsed_time = time.time() - start_time
        total_patches = sum((len(f) for f in features_list))
        patches_per_sec = total_patches / elapsed_time
        images_per_sec = num_images / elapsed_time
        if self.device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1000000000.0
            memory_cached = torch.cuda.max_memory_reserved() / 1000000000.0
        else:
            memory_used = memory_cached = 0
        metrics = {'total_images': num_images, 'total_patches': total_patches, 'elapsed_time': elapsed_time, 'patches_per_second': patches_per_sec, 'images_per_second': images_per_sec, 'memory_used_gb': memory_used, 'memory_cached_gb': memory_cached, 'batch_size': self.batch_size, 'mixed_precision': self.mixed_precision}
        return metrics

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def get_optimal_batch_size(self, target_memory_gb: float=30.0) -> int:
        if self.device.type != 'cuda':
            return self.batch_size
        patch_memory_mb = 1.5
        if self.model_type.startswith('resnet'):
            patch_memory_mb = 1.0
        target_memory_mb = target_memory_gb * 1000
        optimal_batch_size = int(target_memory_mb / patch_memory_mb)
        optimal_batch_size = optimal_batch_size // 8 * 8
        optimal_batch_size = max(8, min(optimal_batch_size, 512))
        return optimal_batch_size

def test_optimized_extractor():
    extractor = OptimizedCNNFeatureExtractor(model_type='vgg16', batch_size=64, mixed_precision=True, use_tf32=True)
    metrics = extractor.benchmark_performance(image_size=(512, 512), num_images=3)
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    features, coords = extractor.extract_single_image_optimized(sample_image)
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1000000000.0
