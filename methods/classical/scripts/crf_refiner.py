#!/usr/bin/env python3
import numpy as np
import cv2
import logging
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import time
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
logger = logging.getLogger(__name__)

class DenseCRFRefiner:

    def __init__(self, config: Dict):
        if not CRF_AVAILABLE:
            raise ImportError('pydensecrf is required. Install with: pip install pydensecrf')
        self.config = config
        self.bilateral_sxy = config.get('bilateral_sxy', [80, 80])
        self.bilateral_srgb = config.get('bilateral_srgb', [13, 13, 13])
        self.gaussian_sxy = config.get('gaussian_sxy', [3, 3])
        self.inference_steps = config.get('inference_steps', 5)
        self.compat_bilateral = config.get('compat_bilateral', 10)
        self.compat_gaussian = config.get('compat_gaussian', 3)

    def apply_dense_crf(self, image: np.ndarray, probabilities: np.ndarray, return_probabilities: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if not CRF_AVAILABLE:
            if probabilities.ndim == 3:
                return np.argmax(probabilities, axis=2)
            else:
                h, w = image.shape[:2]
                return np.argmax(probabilities, axis=1).reshape(h, w)
        start_time = time.time()
        h, w = image.shape[:2]
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        image = np.ascontiguousarray(image)
        if probabilities.ndim == 3:
            num_classes = probabilities.shape[2]
            probabilities = probabilities.reshape(h * w, num_classes)
        else:
            num_classes = probabilities.shape[1]
        probabilities = probabilities / (probabilities.sum(axis=1, keepdims=True) + 1e-08)
        probabilities = np.ascontiguousarray(probabilities)
        d = dcrf.DenseCRF2D(w, h, num_classes)
        probabilities_T = np.ascontiguousarray(probabilities.T)
        unary = unary_from_softmax(probabilities_T)
        unary = np.ascontiguousarray(unary)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=tuple(self.bilateral_sxy), srgb=tuple(self.bilateral_srgb), rgbim=image, compat=self.compat_bilateral)
        d.addPairwiseGaussian(sxy=tuple(self.gaussian_sxy), compat=self.compat_gaussian)
        Q = d.inference(self.inference_steps)
        Q = np.array(Q).reshape((num_classes, h, w))
        refined_labels = np.argmax(Q, axis=0)
        processing_time = time.time() - start_time
        if return_probabilities:
            refined_probs = Q.transpose(1, 2, 0)
            return (refined_labels, refined_probs)
        return refined_labels

    def apply_crf_batch(self, images: list, probabilities_list: list, show_progress: bool=True) -> list:
        refined_predictions = []
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(zip(images, probabilities_list), total=len(images), desc='Applying CRF')
        else:
            iterator = zip(images, probabilities_list)
        for image, probabilities in iterator:
            refined = self.apply_dense_crf(image, probabilities)
            refined_predictions.append(refined)
        return refined_predictions

    def compare_predictions(self, image: np.ndarray, original_probs: np.ndarray, verbose: bool=True) -> Dict:
        original_labels = np.argmax(original_probs, axis=-1)
        if original_labels.ndim == 1:
            h, w = image.shape[:2]
            original_labels = original_labels.reshape(h, w)
        refined_labels = self.apply_dense_crf(image, original_probs)
        different_pixels = (original_labels != refined_labels).sum()
        total_pixels = original_labels.size
        change_percentage = different_pixels / total_pixels * 100
        unique_classes = np.unique(np.concatenate([original_labels.flat, refined_labels.flat]))
        class_changes = {}
        for cls in unique_classes:
            orig_count = (original_labels == cls).sum()
            refined_count = (refined_labels == cls).sum()
            class_changes[f'class_{cls}'] = {'original': int(orig_count), 'refined': int(refined_count), 'change': int(refined_count - orig_count)}
        results = {'pixels_changed': int(different_pixels), 'total_pixels': int(total_pixels), 'change_percentage': float(change_percentage), 'class_changes': class_changes}
        if verbose:
            for cls_name, changes in class_changes.items():
                pass
        return results

def create_default_crf_config(num_classes: int=2) -> Dict:
    if num_classes == 2:
        return {'bilateral_sxy': [80, 80], 'bilateral_srgb': [13, 13, 13], 'gaussian_sxy': [3, 3], 'inference_steps': 5, 'compat_bilateral': 10, 'compat_gaussian': 3}
    else:
        return {'bilateral_sxy': [80, 80], 'bilateral_srgb': [10, 10, 10], 'gaussian_sxy': [5, 5], 'inference_steps': 10, 'compat_bilateral': 15, 'compat_gaussian': 5}

def refine_xgboost_predictions(model_predictions: np.ndarray, images: list, crf_config: Optional[Dict]=None, num_classes: int=2) -> list:
    if crf_config is None:
        crf_config = create_default_crf_config(num_classes)
    refiner = DenseCRFRefiner(crf_config)
    return refiner.apply_crf_batch(images, model_predictions)
