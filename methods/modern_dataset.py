#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2

class ModernSAMDataset(Dataset):

    def __init__(self, coco_json: str, data_root: str, prompt_policy: str='P1', img_size: int=1024, augment: bool=True, jitter_pixels: int=5, color_jitter: float=0.1, split: str='train'):
        self.coco_json = coco_json
        self.data_root = Path(data_root)
        self.prompt_policy = prompt_policy
        self.img_size = img_size
        self.augment = augment
        self.jitter_pixels = jitter_pixels
        self.split = split
        with open(coco_json, 'r') as f:
            self.coco_data = json.load(f)
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.samples = []
        annotations = self.coco_data.get('annotations')
        if annotations:
            ann_by_image: Dict[Union[str, int], Dict] = {}
            for ann in annotations:
                img_id = ann.get('image_id')
                if img_id is not None and img_id not in ann_by_image:
                    ann_by_image[img_id] = ann
            for img_id, img_info in self.images.items():
                ann = ann_by_image.get(img_id)
                if ann is not None:
                    self.samples.append({'image_info': img_info, 'annotation': ann})
        else:
            for img_info in self.images.values():
                self.samples.append({'image_info': img_info, 'annotation': {}})
        if augment and color_jitter > 0:
            self.color_transform = T.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=color_jitter / 2)
        else:
            self.color_transform = T.Lambda(lambda x: x)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_info = sample['image_info']
        annotation = sample['annotation']
        img_file = str(img_info['file_name']).replace('\\', '/').lstrip('./')
        image_path = self.data_root / 'composites' / img_file
        mask_file_override = sample['image_info'].get('mask_file') or img_info.get('mask_file')
        if mask_file_override:
            mask_rel = str(mask_file_override).replace('\\', '/').lstrip('./')
            mask_path = self.data_root / mask_rel
        else:
            parts = img_file.split('/')
            if len(parts) >= 3:
                split_scale_uuid = '/'.join(parts[1:])
                mask_rel = split_scale_uuid.replace('.jpg', '.png')
                mask_path = self.data_root / 'masks' / mask_rel
            else:
                mask_rel = img_file.replace('.jpg', '.png')
                mask_path = self.data_root / 'masks' / mask_rel
        if not image_path.exists():
            raise FileNotFoundError(f"Modern WS2: Image not found: {image_path} (from file_name={img_info['file_name']})")
        if not mask_path.exists():
            raise FileNotFoundError(f'Modern WS2: Mask not found: {mask_path} (hint: set mask_file in JSON if using legacy layout)')
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.augment:
            image, mask = self._apply_augmentations(image, mask)
        image_tensor = self._to_tensor_and_normalize(image)
        mask_tensor = self._mask_to_tensor(mask)
        prompts = self._generate_prompts(mask_tensor, img_info)
        meta = {'corruption': img_info['corruption'], 'scale_bin': img_info['scale_bin'], 'severity': img_info['severity'], 'sample_id': img_info['id'], 'source_coco_id': img_info.get('source_coco_id', -1)}
        return {'image': image_tensor, 'mask': mask_tensor, 'prompts': prompts, 'meta': meta}

    def _apply_augmentations(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.augment and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        image = self.color_transform(image)
        return (image, mask)

    def _to_tensor_and_normalize(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        tensor = T.ToTensor()(image)
        return self.normalize(tensor)

    def _mask_to_tensor(self, mask: Image.Image) -> torch.Tensor:
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_array = np.array(mask)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()
        mask_tensor = (mask_tensor > 0).float()
        return mask_tensor

    def _generate_prompts(self, mask: torch.Tensor, img_info: Dict) -> Dict[str, torch.Tensor]:
        mask_np = mask.squeeze().numpy()
        contours, _ = cv2.findContours((mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            h, w = mask_np.shape
            centroid = (w // 2, h // 2)
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            else:
                h, w = mask_np.shape
                centroid = (w // 2, h // 2)
        if self.augment:
            jitter_x = random.randint(-self.jitter_pixels, self.jitter_pixels)
            jitter_y = random.randint(-self.jitter_pixels, self.jitter_pixels)
            centroid = (max(0, min(mask_np.shape[1] - 1, centroid[0] + jitter_x)), max(0, min(mask_np.shape[0] - 1, centroid[1] + jitter_y)))
        if self.prompt_policy == 'P1':
            points = torch.tensor([[centroid[0], centroid[1]]], dtype=torch.float)
            labels = torch.tensor([1], dtype=torch.long)
            y_indices, x_indices = np.where(mask_np > 0.5)
            if len(y_indices) > 0:
                bbox = [x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()]
                neg_x = max(0, bbox[0] - 20) if random.random() > 0.5 else min(mask_np.shape[1] - 1, bbox[2] + 20)
                neg_y = max(0, bbox[1] - 20) if random.random() > 0.5 else min(mask_np.shape[0] - 1, bbox[3] + 20)
                points = torch.cat([points, torch.tensor([[neg_x, neg_y]], dtype=torch.float)])
                labels = torch.cat([labels, torch.tensor([0], dtype=torch.long)])
        elif self.prompt_policy == 'P2':
            y_indices, x_indices = np.where(mask_np > 0.5)
            if len(y_indices) > 0:
                bbox = [x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()]
                x_coords = np.linspace(bbox[0], bbox[2], 4)
                y_coords = np.linspace(bbox[1], bbox[3], 2)
                grid_points = []
                grid_labels = []
                for y in y_coords:
                    for x in x_coords:
                        x, y = (int(x), int(y))
                        if 0 <= x < mask_np.shape[1] and 0 <= y < mask_np.shape[0]:
                            grid_points.append([x, y])
                            grid_labels.append(1 if mask_np[y, x] > 0.5 else 0)
                points = torch.tensor(grid_points, dtype=torch.float)
                labels = torch.tensor(grid_labels, dtype=torch.long)
            else:
                points = torch.tensor([[centroid[0], centroid[1]]], dtype=torch.float)
                labels = torch.tensor([1], dtype=torch.long)
        return {'points': points, 'labels': labels}

def create_modern_sam_datasets(data_root: str='../ws1_data/data/ws1', train_json: str=None, val_json: str=None, test_json: str=None, prompt_policy: str='P1', img_size: int=1024, train_augment: bool=True, **kwargs) -> Dict[str, ModernSAMDataset]:
    data_root = Path(data_root)
    if train_json is None:
        train_json = data_root / 'train.json'
    if val_json is None:
        val_json = data_root / 'val.json'
    if test_json is None:
        test_json = data_root / 'test.json'
    datasets = {}
    if train_json and Path(train_json).exists():
        datasets['train'] = ModernSAMDataset(coco_json=str(train_json), data_root=str(data_root), prompt_policy=prompt_policy, img_size=img_size, augment=train_augment, split='train', **kwargs)
    if val_json and Path(val_json).exists():
        datasets['val'] = ModernSAMDataset(coco_json=str(val_json), data_root=str(data_root), prompt_policy=prompt_policy, img_size=img_size, augment=False, split='val', **kwargs)
    if test_json and Path(test_json).exists():
        datasets['test'] = ModernSAMDataset(coco_json=str(test_json), data_root=str(data_root), prompt_policy=prompt_policy, img_size=img_size, augment=False, split='test', **kwargs)
    return datasets
