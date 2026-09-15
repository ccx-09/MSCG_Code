#!/usr/bin/env python3
import sys
import json
import yaml
import argparse
import pickle
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
import logging
import time
from tqdm import tqdm
sys.path.append(str(Path(__file__).parent))
import xgboost as xgb
from crf_refiner import DenseCRFRefiner, create_default_crf_config
logger = logging.getLogger(__name__)

class XGBoostCRFInference:

    def __init__(self, model_path: str, config: Dict, preprocessor_path: Optional[str]=None, feature_method: str='handcrafted', cnn_model: str='vgg16', patch_size: int=64, stride: int=32, batch_size: int=64, num_workers: int=4):
        self.model_path = Path(model_path)
        self.config = config
        self.preprocessor_path = preprocessor_path
        self.feature_method = feature_method
        self.cnn_model = cnn_model
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))
        self.preprocessor = None
        if preprocessor_path and Path(preprocessor_path).exists():
            try:
                from preprocessing_pipeline import ClassicalPreprocessor
                self.preprocessor = ClassicalPreprocessor.load(preprocessor_path)
            except Exception:
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
        self.reverse_mapping = None
        try:
            run_dir = self.model_path.parent.parent if self.model_path.parent.name == 'EMERGENCY_SAVES' else self.model_path.parent
            rev_path = run_dir / 'reverse_mapping.json'
            if rev_path.exists():
                import json
                raw = json.loads(rev_path.read_text())
                self.reverse_mapping = {int(k): int(v) for k, v in raw.items()}
        except Exception as e:
            pass
        self.feature_extractor = None
        if self.feature_method == 'handcrafted':
            try:
                from feature_extraction import PixelFeatureExtractor
                self.feature_extractor = PixelFeatureExtractor()
            except ImportError:
                self.feature_extractor = None
        elif self.feature_method == 'cnn_gpu':
            try:
                from gpu_optimized_extractor import GPUOptimizedCNNExtractor
                self.feature_extractor = GPUOptimizedCNNExtractor(model_type=self.cnn_model, batch_size=self.batch_size, num_workers=self.num_workers, mixed_precision=True, use_tf32=True)
            except Exception as e:
                self.feature_extractor = None
        self.crf_enabled = config.get('crf', {}).get('enabled', False)
        self.crf_refiner = None
        if self.crf_enabled:
            crf_config = config.get('crf', {})
            num_classes = config.get('num_classes', 2)
            if 'bilateral_sxy' not in crf_config:
                crf_config = create_default_crf_config(num_classes)
            try:
                self.crf_refiner = DenseCRFRefiner(crf_config)
            except ImportError:
                self.crf_enabled = False
        self.dataset_type = config.get('dataset', {}).get('type', 'ws1')
        if hasattr(self.model, 'classes_'):
            self.num_classes = len(self.model.classes_)
        elif hasattr(self.model, '_class_count'):
            self.num_classes = self.model._class_count
        else:
            try:
                sample_features = np.zeros((1, 4096 if feature_method == 'cnn_gpu' else 56))
                sample_pred = self.model.predict_proba(sample_features)
                self.num_classes = sample_pred.shape[1]
            except:
                self.num_classes = config.get('dataset', {}).get('num_classes', 2)

    def extract_dense_features(self, image: np.ndarray) -> np.ndarray:
        if self.feature_extractor:
            feature_map, _feature_names = self.feature_extractor.extract_all_features(image)
            return feature_map
        else:
            h, w = image.shape[:2]
            img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            loc_features = np.stack([y_coords / h, x_coords / w], axis=-1)
            color_features = img_lab.astype(np.float32) / 255.0
            grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_features = np.stack([grad_x, grad_y], axis=-1)
            laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
            texture_features = laplacian.reshape(h, w, 1)
            feature_map = np.concatenate([loc_features, color_features, gradient_features, texture_features], axis=-1)
            return feature_map

    def predict_image(self, image: np.ndarray, return_probabilities: bool=True) -> Dict:
        start_time = time.time()
        h, w = image.shape[:2]
        if self.feature_method == 'handcrafted':
            feature_map = self.extract_dense_features(image)
            features_flat = feature_map.reshape(h * w, -1)
            if self.preprocessor:
                features_flat = self.preprocessor.transform(features_flat)
            if return_probabilities:
                probabilities_flat = self.model.predict_proba(features_flat)
                probabilities = probabilities_flat.reshape(h, w, self.num_classes)
                raw_predictions = np.argmax(probabilities, axis=2)
            else:
                raw_predictions = self.model.predict(features_flat).reshape(h, w)
                probabilities = None
        elif self.feature_method == 'cnn_gpu':
            if self.feature_extractor is None:
                raise RuntimeError('CNN feature extractor is not initialized')
            features, coords = self.feature_extractor.extract_single_image(image, patch_size=self.patch_size, stride=self.stride)
            if features.size == 0:
                raise RuntimeError('No CNN features extracted from image')
            if self.preprocessor is not None:
                features = self.preprocessor.transform(features)
            probs_patch = self.model.predict_proba(features)
            prob_sum = np.zeros((h, w, self.num_classes), dtype=np.float32)
            count = np.zeros((h, w), dtype=np.float32)
            ps = int(self.patch_size)
            coords_int = coords.astype(int)
            for (cy, cx), pvec in zip(coords_int, probs_patch):
                y0 = int(cy - ps // 2)
                x0 = int(cx - ps // 2)
                y1 = y0 + ps
                x1 = x0 + ps
                if y0 < 0:
                    y0 = 0
                if x0 < 0:
                    x0 = 0
                if y1 > h:
                    y1 = h
                if x1 > w:
                    x1 = w
                if y0 >= y1 or x0 >= x1:
                    continue
                prob_sum[y0:y1, x0:x1, :] += pvec[None, None, :]
                count[y0:y1, x0:x1] += 1.0
            count = np.maximum(count, 1e-06)
            probabilities = prob_sum / count[:, :, None]
            raw_predictions = np.argmax(probabilities, axis=2)
        else:
            raise ValueError(f'Unsupported feature method: {self.feature_method}')
        refined_predictions = None
        if self.crf_enabled and probabilities is not None:
            try:
                rgb_for_crf = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                refined_predictions = self.crf_refiner.apply_dense_crf(rgb_for_crf, probabilities)
            except Exception as e:
                refined_predictions = raw_predictions
        prediction_time = time.time() - start_time
        results = {'raw_predictions': raw_predictions, 'refined_predictions': refined_predictions, 'probabilities': probabilities, 'image_shape': (h, w), 'prediction_time': prediction_time, 'crf_applied': self.crf_enabled and refined_predictions is not None}
        return results

    def predict_batch(self, images: List[np.ndarray], show_progress: bool=True) -> List[Dict]:
        results = []
        iterator = tqdm(images, desc='Running inference') if show_progress else images
        for image in iterator:
            result = self.predict_image(image)
            results.append(result)
        return results

    def evaluate_predictions(self, predictions: List[Dict], ground_truth_masks: List[np.ndarray], use_crf: bool=True) -> Dict:
        assert len(predictions) == len(ground_truth_masks), 'Predictions and ground truth must have same length'
        mious = []
        dices = []
        accuracies = []
        for pred_dict, gt_mask in zip(predictions, ground_truth_masks):
            if use_crf and pred_dict['crf_applied']:
                pred_mask = pred_dict['refined_predictions']
            else:
                pred_mask = pred_dict['raw_predictions']
            miou = self._calculate_miou(pred_mask, gt_mask)
            dice = self._calculate_dice(pred_mask, gt_mask)
            accuracy = self._calculate_accuracy(pred_mask, gt_mask)
            mious.append(miou)
            dices.append(dice)
            accuracies.append(accuracy)
        results = {'mean_miou': float(np.mean(mious)), 'mean_dice': float(np.mean(dices)), 'mean_accuracy': float(np.mean(accuracies)), 'std_miou': float(np.std(mious)), 'std_dice': float(np.std(dices)), 'std_accuracy': float(np.std(accuracies)), 'individual_mious': mious, 'num_images': len(predictions), 'crf_used': use_crf}
        return results

    def _calculate_miou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        if self.num_classes == 2:
            pred_bin = (pred_mask > 0).astype(np.uint8)
            gt_bin = (gt_mask > 0).astype(np.uint8)
            intersection = (pred_bin & gt_bin).sum()
            union = (pred_bin | gt_bin).sum()
            return float(intersection / union) if union > 0 else 1.0
        ious = []
        for cls in range(self.num_classes):
            pred_cls = pred_mask == cls
            gt_cls = gt_mask == cls
            intersection = (pred_cls & gt_cls).sum()
            union = (pred_cls | gt_cls).sum()
            if union > 0:
                ious.append(intersection / union)
        return float(np.mean(ious)) if ious else 0.0

    def _calculate_per_class_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[int, float]:
        per_class_iou = {}
        if self.num_classes == 2:
            pred_bin = (pred_mask > 0).astype(np.uint8)
            gt_bin = (gt_mask > 0).astype(np.uint8)
            pred_bg = pred_bin == 0
            gt_bg = gt_bin == 0
            intersection_bg = (pred_bg & gt_bg).sum()
            union_bg = (pred_bg | gt_bg).sum()
            per_class_iou[0] = float(intersection_bg / union_bg) if union_bg > 0 else 1.0
            pred_fg = pred_bin == 1
            gt_fg = gt_bin == 1
            intersection_fg = (pred_fg & gt_fg).sum()
            union_fg = (pred_fg | gt_fg).sum()
            per_class_iou[1] = float(intersection_fg / union_fg) if union_fg > 0 else 1.0
        else:
            for cls in range(self.num_classes):
                pred_cls = pred_mask == cls
                gt_cls = gt_mask == cls
                intersection = (pred_cls & gt_cls).sum()
                union = (pred_cls | gt_cls).sum()
                if union > 0:
                    per_class_iou[cls] = float(intersection / union)
                else:
                    per_class_iou[cls] = 0.0
        return per_class_iou

    def _calculate_dice(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        if self.num_classes == 2:
            pred_bin = (pred_mask > 0).astype(np.uint8)
            gt_bin = (gt_mask > 0).astype(np.uint8)
            intersection = (pred_bin & gt_bin).sum()
            total = pred_bin.sum() + gt_bin.sum()
            return float(2 * intersection / total) if total > 0 else 1.0
        dices = []
        for cls in range(self.num_classes):
            pred_cls = pred_mask == cls
            gt_cls = gt_mask == cls
            intersection = (pred_cls & gt_cls).sum()
            total = pred_cls.sum() + gt_cls.sum()
            if total > 0:
                dices.append(2 * intersection / total)
        return float(np.mean(dices)) if dices else 0.0

    def _calculate_accuracy(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        correct = (pred_mask == gt_mask).sum()
        total = gt_mask.size
        return float(correct / total)

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--preprocessor')
    parser.add_argument('--output')
    parser.add_argument('--dataset', choices=['ws1'], default='ws1')
    parser.add_argument('--test-images', type=int)
    parser.add_argument('--feature-method', choices=['handcrafted', 'cnn_gpu'], default='cnn_gpu')
    parser.add_argument('--cnn-model', choices=['vgg16', 'resnet50', 'resnet101'], default='vgg16')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--ws1-root', default='../ws1_data/data/ws1')
    parser.add_argument('--ws1-json', default=None)
    parser.add_argument('--compare-crf', action='store_true')
    parser.add_argument('--visual-samples', type=int, default=20)
    args = parser.parse_args()
    config = load_config(args.config)
    config['dataset']['type'] = args.dataset
    config['dataset']['num_classes'] = 2
    dataset_type = args.dataset
    config['dataset']['type'] = dataset_type
    inference = XGBoostCRFInference(model_path=args.model, config=config, preprocessor_path=args.preprocessor, feature_method=args.feature_method, cnn_model=args.cnn_model, patch_size=args.patch_size, stride=args.stride, batch_size=args.batch_size, num_workers=args.n_workers)
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path('inference_results')
        output_dir.mkdir(exist_ok=True)
    try:
        ws1_root = Path(args.ws1_root)
        split_json = Path(args.ws1_json) if args.ws1_json else ws1_root / 'splits' / f'{args.split}.json'
        if not split_json.exists():
            raise FileNotFoundError(f'WS1 split file not found: {split_json}')
        with open(split_json, 'r') as f:
            split_data = json.load(f)
        images_entries = split_data.get('images', [])
        if not images_entries:
            raise RuntimeError('No images found in WS1 split JSON')
        image_mask_pairs = []
        for rec in images_entries:
            file_name = str(rec['file_name']).replace('\\', '/').lstrip('./')
            if file_name.startswith('composites/'):
                img_path = ws1_root / file_name
            else:
                img_path = ws1_root / 'composites' / file_name
            mask_file = rec.get('mask_file')
            if mask_file:
                msk_path = ws1_root / str(mask_file).replace('\\', '/').lstrip('./')
            else:
                parts = file_name.split('/')
                if len(parts) >= 4:
                    _, split, scale_bin, filename = (parts[0], parts[1], parts[2], parts[3])
                    mask_filename = filename.replace('.jpg', '.png')
                    msk_path = ws1_root / 'masks' / split / scale_bin / mask_filename
                else:
                    msk_path = ws1_root / 'masks' / file_name.replace('.jpg', '.png')
            if img_path.exists() and msk_path.exists():
                image_mask_pairs.append((str(img_path), str(msk_path)))
        if args.test_images:
            image_mask_pairs = image_mask_pairs[:args.test_images]
        if not image_mask_pairs:
            raise RuntimeError('No valid WS1 image/mask pairs to run inference')
        per_image_results = []
        corruption_stats = {}
        severity_stats = {}
        scale_stats = {}
        all_per_class_ious = []
        for idx, (img_path, msk_path) in enumerate(tqdm(image_mask_pairs, desc='Dense inference')):
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            gt_mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                continue
            path_parts = Path(img_path).parts
            corruption = 'unknown'
            scale_bin = 'unknown'
            severity = 'unknown'
            if len(path_parts) >= 4:
                corruption = path_parts[-4]
                scale_severity = path_parts[-2]
                if '_k' in scale_severity:
                    scale_bin, severity_str = scale_severity.split('_k')
                    severity = int(severity_str)
            if inference.num_classes == 2:
                gt_eval = (gt_mask > 0).astype(np.uint8)
            else:
                gt_eval = gt_mask.astype(np.int32)
            pred_dict = inference.predict_image(image_bgr, return_probabilities=True)
            raw_pred = pred_dict['raw_predictions']
            crf_pred = pred_dict['refined_predictions'] if pred_dict['crf_applied'] else None
            pred_mask = crf_pred if crf_pred is not None else raw_pred
            miou = inference._calculate_miou(pred_mask, gt_eval)
            dice = inference._calculate_dice(pred_mask, gt_eval)
            acc = inference._calculate_accuracy(pred_mask, gt_eval)
            per_class_iou = inference._calculate_per_class_iou(pred_mask, gt_eval)
            all_per_class_ious.append(per_class_iou)
            crf_delta = 0.0
            result = {'image_path': img_path, 'mask_path': msk_path, 'corruption': corruption, 'scale_bin': scale_bin, 'severity': severity, 'miou': float(miou), 'dice': float(dice), 'accuracy': float(acc), 'per_class_iou': per_class_iou, 'crf_delta': float(crf_delta), 'height': int(image_bgr.shape[0]), 'width': int(image_bgr.shape[1])}
            per_image_results.append(result)
            for stats_dict, key in [(corruption_stats, corruption), (severity_stats, severity), (scale_stats, scale_bin)]:
                if key not in stats_dict:
                    stats_dict[key] = {'miou': [], 'dice': [], 'accuracy': [], 'count': 0}
                stats_dict[key]['miou'].append(miou)
                stats_dict[key]['dice'].append(dice)
                stats_dict[key]['accuracy'].append(acc)
                stats_dict[key]['count'] += 1
            out_name = Path(img_path).stem + ('_crf.png' if pred_dict['crf_applied'] else '_raw.png')
            pred_out = Path(output_dir) / out_name
            if inference.num_classes == 2:
                cv2.imwrite(str(pred_out), pred_mask.astype(np.uint8))
            else:
                cv2.imwrite(str(pred_out), pred_mask.astype(np.uint8))
        if per_image_results:
            mean_miou = float(np.mean([r['miou'] for r in per_image_results]))
            mean_dice = float(np.mean([r['dice'] for r in per_image_results]))
            mean_acc = float(np.mean([r['accuracy'] for r in per_image_results]))
        else:
            mean_miou = mean_dice = mean_acc = 0.0

        def compute_summary_stats(stats_dict):
            summary = {}
            for key, metrics in stats_dict.items():
                if metrics['count'] > 0:
                    summary[key] = {'count': metrics['count'], 'mean_miou': float(np.mean(metrics['miou'])), 'std_miou': float(np.std(metrics['miou'])), 'mean_dice': float(np.mean(metrics['dice'])), 'std_dice': float(np.std(metrics['dice'])), 'mean_accuracy': float(np.mean(metrics['accuracy'])), 'std_accuracy': float(np.std(metrics['accuracy']))}
            return summary
        corruption_summary = compute_summary_stats(corruption_stats)
        severity_summary = compute_summary_stats(severity_stats)
        scale_summary = compute_summary_stats(scale_stats)
        per_class_summary = {}
        if all_per_class_ious:
            for cls in range(inference.num_classes):
                cls_ious = [iou_dict.get(cls, 0.0) for iou_dict in all_per_class_ious if cls in iou_dict]
                if cls_ious:
                    per_class_summary[cls] = {'mean_iou': float(np.mean(cls_ious)), 'std_iou': float(np.std(cls_ious)), 'count': len(cls_ious)}
        results_summary = {'dataset_type': dataset_type, 'split': args.split, 'feature_method': args.feature_method, 'cnn_model': args.cnn_model if args.feature_method == 'cnn_gpu' else None, 'num_classes_detected': inference.num_classes, 'num_images': len(per_image_results), 'mean_miou': mean_miou, 'mean_dice': mean_dice, 'mean_accuracy': mean_acc, 'per_class_iou': per_class_summary, 'per_corruption': corruption_summary, 'per_severity': severity_summary, 'per_scale': scale_summary, 'per_image': per_image_results, 'crf_enabled': inference.crf_enabled}
        with open(output_dir / 'inference_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        if dataset_type == 'ws1' and per_image_results:
            import pandas as pd
            df_images = pd.DataFrame(per_image_results)
            df_images.to_csv(output_dir / 'per_image_results.csv', index=False)
            summary_rows = []
            for corruption, stats in corruption_summary.items():
                summary_rows.append({'category': 'corruption', 'value': corruption, 'count': stats['count'], 'mean_miou': stats['mean_miou'], 'std_miou': stats['std_miou'], 'mean_dice': stats['mean_dice'], 'std_dice': stats['std_dice'], 'mean_accuracy': stats['mean_accuracy'], 'std_accuracy': stats['std_accuracy']})
            for severity, stats in severity_summary.items():
                summary_rows.append({'category': 'severity', 'value': str(severity), 'count': stats['count'], 'mean_miou': stats['mean_miou'], 'std_miou': stats['std_miou'], 'mean_dice': stats['mean_dice'], 'std_dice': stats['std_dice'], 'mean_accuracy': stats['mean_accuracy'], 'std_accuracy': stats['std_accuracy']})
            for scale, stats in scale_summary.items():
                summary_rows.append({'category': 'scale_bin', 'value': scale, 'count': stats['count'], 'mean_miou': stats['mean_miou'], 'std_miou': stats['std_miou'], 'mean_dice': stats['mean_dice'], 'std_dice': stats['std_dice'], 'mean_accuracy': stats['mean_accuracy'], 'std_accuracy': stats['std_accuracy']})
            summary_rows.append({'category': 'overall', 'value': 'all', 'count': len(per_image_results), 'mean_miou': mean_miou, 'std_miou': float(np.std([r['miou'] for r in per_image_results])), 'mean_dice': mean_dice, 'std_dice': float(np.std([r['dice'] for r in per_image_results])), 'mean_accuracy': mean_acc, 'std_accuracy': float(np.std([r['accuracy'] for r in per_image_results]))})
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_csv(output_dir / 'summary_statistics.csv', index=False)
        config_summary = {'model_path': str(args.model), 'config_path': str(args.config), 'feature_method': args.feature_method, 'patch_size': args.patch_size, 'stride': args.stride, 'crf_enabled': inference.crf_enabled, 'num_classes_detected': inference.num_classes, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
        with open(output_dir / 'inference_config.json', 'w') as f:
            json.dump(config_summary, f, indent=2)
        if dataset_type == 'ws1':
            if per_class_summary:
                for cls, stats in per_class_summary.items():
                    pass
            for corruption, stats in corruption_summary.items():
                pass
            for severity, stats in severity_summary.items():
                pass
            for scale, stats in scale_summary.items():
                pass
            if inference.num_classes > 2:
                pass
    except Exception as e:
        error_summary = {'dataset_type': dataset_type, 'error': str(e), 'status': 'failed'}
        with open(output_dir / 'inference_results.json', 'w') as f:
            json.dump(error_summary, f, indent=2)
        raise
if __name__ == '__main__':
    main()
