#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skimage import feature, measure, segmentation
from skimage.filters import sobel, gaussian, laplace
from skimage.morphology import disk, local_maxima, local_minima
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
import json
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)

class PixelFeatureExtractor:

    def __init__(self, patch_size: int=7, n_orientations: int=8, gabor_frequencies: List[float]=[0.1, 0.3, 0.5], lbp_radius: int=3, lbp_n_points: int=24):
        self.patch_size = patch_size
        self.n_orientations = n_orientations
        self.gabor_frequencies = gabor_frequencies
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self._gabor_kernels = self._create_gabor_filters()
        self.disk_small = disk(1)
        self.disk_medium = disk(3)
        self.disk_large = disk(5)

    def _create_gabor_filters(self) -> List[np.ndarray]:
        kernels = []
        for freq in self.gabor_frequencies:
            for angle in np.linspace(0, np.pi, self.n_orientations, endpoint=False):
                kernel = cv2.getGaborKernel((21, 21), sigma=3, theta=angle, lambd=1.0 / freq, gamma=0.5, psi=0, ktype=cv2.CV_32F)
                kernels.append(kernel)
        return kernels

    def extract_color_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        features = {}
        h, w = image.shape[:2]
        features['r'] = image[:, :, 0]
        features['g'] = image[:, :, 1]
        features['b'] = image[:, :, 2]
        features['rgb_intensity'] = np.mean(image, axis=2)
        features['rgb_std'] = np.std(image, axis=2)
        hsv = rgb2hsv(image)
        features['h'] = hsv[:, :, 0]
        features['s'] = hsv[:, :, 1]
        features['v'] = hsv[:, :, 2]
        lab = rgb2lab(image)
        features['l'] = lab[:, :, 0] / 100.0
        features['a'] = (lab[:, :, 1] + 87) / 185.0
        features['lab_b'] = (lab[:, :, 2] + 108) / 202.0
        return features

    def extract_texture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        if gray.max() > 1:
            gray = gray.astype(np.float32) / 255.0
        features = {}
        lbp = feature.local_binary_pattern(gray, self.lbp_n_points, self.lbp_radius, method='uniform')
        features['lbp'] = lbp / self.lbp_n_points
        for i, kernel in enumerate(self._gabor_kernels):
            response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            features[f'gabor_{i}'] = np.abs(response)
        from skimage.filters.rank import mean
        gray_uint8 = (gray * 255).astype(np.uint8)
        features['texture_std'] = mean(gray_uint8, disk(3)) / 255.0
        return features

    def extract_gradient_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        if gray.max() > 1:
            gray = gray.astype(np.float32) / 255.0
        features = {}
        sobel_h = sobel(gray, axis=0)
        sobel_v = sobel(gray, axis=1)
        features['sobel_h'] = np.abs(sobel_h)
        features['sobel_v'] = np.abs(sobel_v)
        features['sobel_mag'] = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
        features['sobel_dir'] = np.arctan2(sobel_v, sobel_h) / (2 * np.pi) + 0.5
        features['laplacian'] = np.abs(laplace(gray))
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['grad_mag'] = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return features

    def extract_morphological_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        if gray.max() > 1:
            gray = gray.astype(np.float32) / 255.0
        features = {}
        gray_uint8 = (gray * 255).astype(np.uint8)
        features['erosion_small'] = cv2.morphologyEx(gray_uint8, cv2.MORPH_ERODE, self.disk_small) / 255.0
        features['dilation_small'] = cv2.morphologyEx(gray_uint8, cv2.MORPH_DILATE, self.disk_small) / 255.0
        features['opening'] = cv2.morphologyEx(gray_uint8, cv2.MORPH_OPEN, self.disk_medium) / 255.0
        features['closing'] = cv2.morphologyEx(gray_uint8, cv2.MORPH_CLOSE, self.disk_medium) / 255.0
        features['tophat'] = cv2.morphologyEx(gray_uint8, cv2.MORPH_TOPHAT, self.disk_large) / 255.0
        features['blackhat'] = cv2.morphologyEx(gray_uint8, cv2.MORPH_BLACKHAT, self.disk_large) / 255.0
        return features

    def extract_spatial_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        h, w = image.shape[:2]
        features = {}
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        features['x_coord'] = x_coords.astype(np.float32) / w
        features['y_coord'] = y_coords.astype(np.float32) / h
        center_x, center_y = (w // 2, h // 2)
        features['dist_center'] = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) / np.sqrt(center_x ** 2 + center_y ** 2)
        features['dist_tl'] = np.sqrt(x_coords ** 2 + y_coords ** 2) / np.sqrt(w ** 2 + h ** 2)
        features['dist_tr'] = np.sqrt((x_coords - w) ** 2 + y_coords ** 2) / np.sqrt(w ** 2 + h ** 2)
        features['dist_bl'] = np.sqrt(x_coords ** 2 + (y_coords - h) ** 2) / np.sqrt(w ** 2 + h ** 2)
        features['dist_br'] = np.sqrt((x_coords - w) ** 2 + (y_coords - h) ** 2) / np.sqrt(w ** 2 + h ** 2)
        return features

    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        color_features = self.extract_color_features(image)
        texture_features = self.extract_texture_features(image)
        gradient_features = self.extract_gradient_features(image)
        morphological_features = self.extract_morphological_features(image)
        spatial_features = self.extract_spatial_features(image)
        all_features = {}
        all_features.update(color_features)
        all_features.update(texture_features)
        all_features.update(gradient_features)
        all_features.update(morphological_features)
        all_features.update(spatial_features)
        feature_list = []
        feature_names = []
        for name, feat_map in all_features.items():
            feature_list.append(feat_map)
            feature_names.append(name)
        feature_array = np.stack(feature_list, axis=-1)
        return (feature_array, feature_names)

    def process_sample_coordinates(self, image: np.ndarray, coordinates: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        feature_array, feature_names = self.extract_all_features(image)
        feature_samples = []
        for y, x in coordinates:
            if 0 <= y < feature_array.shape[0] and 0 <= x < feature_array.shape[1]:
                feature_samples.append(feature_array[y, x, :])
            else:
                feature_samples.append(np.zeros(feature_array.shape[2]))
        features = np.array(feature_samples)
        return (features, feature_names)

    def extract_features_for_multiclass(self, image: np.ndarray, num_classes: int=21) -> np.ndarray:
        if num_classes > 2:
            return self.extract_all_features(image)
        else:
            return self.extract_all_features(image)

    def extract_from_manifest(self, manifest: Union[str, List[Dict]], sampling_rate: float=0.2, max_images: Optional[int]=None, output_prefix: Optional[str]=None, max_samples: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        if isinstance(manifest, str):
            with open(manifest, 'r') as f:
                entries = json.load(f)
        else:
            entries = manifest
        features_list = []
        labels_list = []
        metadata_list: List[Dict] = []
        total_samples = 0
        num_images = len(entries) if max_images is None else min(len(entries), max_images)
        for idx, rec in enumerate(tqdm(entries[:num_images], desc='Extracting from manifest')):
            img_path = rec['image_path']
            msk_path = rec['mask_path']
            if not Path(img_path).exists() or not Path(msk_path).exists():
                continue
            image = np.array(Image.open(img_path).convert('RGB'))
            mask = np.array(Image.open(msk_path))
            feature_array, feature_names = self.extract_all_features(image)
            h, w = mask.shape[:2]
            n_pixels = h * w
            n_sample = int(n_pixels * sampling_rate)
            if max_samples is not None and total_samples + n_sample > max_samples:
                n_sample = max(0, max_samples - total_samples)
            if n_sample == 0:
                break
            flat_idx = np.random.choice(n_pixels, n_sample, replace=False)
            yy = flat_idx // w
            xx = flat_idx % w
            feats = feature_array[yy, xx, :]
            lbls = mask[yy, xx]
            valid = lbls != 255
            feats = feats[valid]
            lbls = lbls[valid].astype(np.int64)
            if feats.size == 0:
                continue
            features_list.append(feats)
            labels_list.append(lbls)
            total_samples += feats.shape[0]
            metadata_list.append({'sample_id': rec.get('sample_id'), 'image_path': img_path, 'mask_path': msk_path, 'num_sampled': int(feats.shape[0])})
            if max_samples is not None and total_samples >= max_samples:
                break
        if not features_list:
            raise RuntimeError('No features extracted from manifest entries')
        features = np.vstack(features_list)
        labels = np.hstack(labels_list)
        if output_prefix:
            output_path = Path(output_prefix).with_suffix('.npz')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, features=features, labels=labels, metadata=metadata_list)
        return (features, labels, metadata_list)

def extract_features_for_sampling_index(sampling_csv: str, ws1_root: str, output_path: str, n_samples: Optional[int]=None) -> None:
    df = pd.read_csv(sampling_csv)
    if n_samples:
        df = df.head(n_samples)
    extractor = PixelFeatureExtractor()
    features_list = []
    labels_list = []
    metadata_list = []
    current_image = None
    current_image_path = None
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Extracting features'):
        image_path = Path(ws1_root) / 'composites' / row['corruption'] / 'train' / f"s{row['scale_bin']}_k{row['severity']}" / f"{row['uuid']}.jpg"
        if str(image_path) != current_image_path:
            if image_path.exists():
                current_image = np.array(Image.open(image_path))
                current_image_path = str(image_path)
            else:
                continue
        coordinates = [(row['y'], row['x'])]
        features, feature_names = extractor.process_sample_coordinates(current_image, coordinates)
        features_list.append(features[0])
        labels_list.append(row['label'])
        metadata_list.append({'uuid': row['uuid'], 'scale_bin': row['scale_bin'], 'severity': row['severity'], 'corruption': row['corruption'], 'x': row['x'], 'y': row['y']})
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, features=features_array, labels=labels_array, feature_names=feature_names, metadata=metadata_list)
    stats = {'n_samples': len(features_array), 'n_features': len(feature_names), 'feature_names': feature_names, 'label_distribution': {'foreground': int(np.sum(labels_array == 1)), 'background': int(np.sum(labels_array == 0))}, 'feature_stats': {'mean': features_array.mean(axis=0).tolist(), 'std': features_array.std(axis=0).tolist(), 'min': features_array.min(axis=0).tolist(), 'max': features_array.max(axis=0).tolist()}}
    stats_path = output_path.with_suffix('.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_csv', required=True)
    parser.add_argument('--ws1_root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n_samples', type=int)
    args = parser.parse_args()
    extract_features_for_sampling_index(args.sampling_csv, args.ws1_root, args.output, args.n_samples)
