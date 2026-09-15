#!/usr/bin/env python3
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict
import uuid as uuid_module
logger = logging.getLogger(__name__)

class VGG16FeatureExtractor:

    def __init__(self, device='cuda', initial_batch_size=2048):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = initial_batch_size
        vgg = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*list(vgg.features.children()))
        self.model.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        self.model.eval()
        self.model.to(self.device)
        if self.device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)

    def extract_patches(self, image: np.ndarray, patch_size: int=64, stride: int=32) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        h, w = image.shape[:2]
        patches = []
        coords = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y + patch_size, x:x + patch_size]
                patches.append(patch)
                coords.append((x, y))
        return (patches, coords)

    def extract_labels(self, mask: np.ndarray, coords: List[Tuple[int, int]], patch_size: int=64) -> np.ndarray:
        labels = []
        for x, y in coords:
            center_x = x + patch_size // 2
            center_y = y + patch_size // 2
            label = 1 if mask[center_y, center_x] > 0 else 0
            labels.append(label)
        return np.array(labels, dtype=np.int32)

    def process_batch_with_fallback(self, patches: List[np.ndarray]) -> np.ndarray:
        while self.batch_size > 32:
            try:
                return self._process_batch(patches, self.batch_size)
            except torch.cuda.OutOfMemoryError:
                self.batch_size = self.batch_size // 2
                torch.cuda.empty_cache()
        return self._process_batch(patches, self.batch_size)

    def _process_batch(self, patches: List[np.ndarray], batch_size: int) -> np.ndarray:
        features_list = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i + batch_size]
                batch_np = np.stack(batch, axis=0)
                batch_tensor = torch.from_numpy(batch_np).float().to(self.device)
                batch_tensor = batch_tensor.permute(0, 3, 1, 2) / 255.0
                batch_tensor = F.interpolate(batch_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                batch_tensor = (batch_tensor - self.mean) / self.std
                if self.device.type == 'cuda':
                    batch_tensor = batch_tensor.to(memory_format=torch.channels_last)
                features = self.model(batch_tensor)
                features = features.squeeze(-1).squeeze(-1)
                features_list.append(features.cpu().numpy())
        return np.vstack(features_list)

def process_fold(args):
    with open(args.fold_manifest, 'r') as f:
        manifest = json.load(f)
    images = manifest['images']
    fold_info = manifest.get('info', {})
    fold_id = fold_info.get('fold', 0)
    split = fold_info.get('split', 'unknown')
    extractor = VGG16FeatureExtractor(device='cuda', initial_batch_size=args.batch_size)
    all_features = []
    all_labels = []
    all_image_indices = []
    all_patch_coords = []
    all_uuids = []
    ws1_root = Path(args.ws1_root)
    composites_dir = ws1_root / 'composites'
    masks_dir = ws1_root / 'masks'
    for img_idx, img_info in enumerate(tqdm(images, desc=f'Fold {fold_id} {split}')):
        img_path = composites_dir / img_info['file_name']
        img_uuid = Path(img_info['file_name']).stem
        mask_file = img_info.get('mask_file')
        if mask_file:
            mask_path = ws1_root / str(mask_file).replace('\\', '/').lstrip('./')
        else:
            path_parts = Path(img_info['file_name']).parts
            mask_filename = Path(img_info['file_name']).stem + '.png'
            if len(path_parts) >= 3:
                img_split = path_parts[1]
                mask_path = masks_dir / img_split / mask_filename
            else:
                mask_path = masks_dir / mask_filename
        try:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
        except Exception as e:
            continue
        patches, coords = extractor.extract_patches(image, args.patch_size, args.stride)
        labels = extractor.extract_labels(mask, coords, args.patch_size)
        features = extractor.process_batch_with_fallback(patches)
        image_indices = np.full(len(patches), img_idx, dtype=np.int32)
        all_features.append(features)
        all_labels.append(labels)
        all_image_indices.append(image_indices)
        all_patch_coords.extend(coords)
        all_uuids.extend([img_uuid] * len(patches))
    final_features = np.vstack(all_features)
    final_labels = np.concatenate(all_labels)
    final_image_indices = np.concatenate(all_image_indices)
    final_patch_coords = np.array(all_patch_coords, dtype=np.int32)
    output_path = Path(args.output_dir) / f'fold{fold_id}_{split}_features.npz'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, features=final_features, labels=final_labels, image_indices=final_image_indices, patch_coords=final_patch_coords, uuid_list=all_uuids, fold=fold_id, split=split, seed=args.seed, n_images=len(images), patch_size=args.patch_size, stride=args.stride)
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_manifest', required=True)
    parser.add_argument('--ws1_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    process_fold(args)
if __name__ == '__main__':
    main()
