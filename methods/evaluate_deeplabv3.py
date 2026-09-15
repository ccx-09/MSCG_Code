#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def load_binary_mask(path, threshold=127):
    img = Image.open(path)
    if img.mode == 'RGB':
        img = img.convert('L')
    mask = np.array(img)
    binary = (mask > threshold).astype(np.uint8)
    return binary

def compute_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / (pred.sum() + gt.sum()) if pred.sum() + gt.sum() > 0 else 0.0
    accuracy = (pred == gt).sum() / len(pred)
    true_positive = intersection
    false_positive = (pred & ~gt).sum()
    false_negative = (gt & ~pred).sum()
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    return {'iou': float(iou), 'dice': float(dice), 'accuracy': float(accuracy), 'precision': float(precision), 'recall': float(recall)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--pred_dir', type=Path, required=True)
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--test_json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with open(args.test_json) as f:
        test_data = json.load(f)
    id_to_mask = {}
    id_to_uuid = {}
    for img_info in test_data['images']:
        img_id = img_info['id']
        mask_file = img_info.get('mask_file')
        if mask_file:
            id_to_mask[str(img_id)] = mask_file
            file_name = img_info.get('file_name', '')
            if file_name:
                uuid = Path(file_name).stem
                id_to_uuid[str(img_id)] = uuid
    all_metrics = []
    missing_gt = 0
    missing_pred = 0
    for img_id, mask_file in tqdm(id_to_mask.items(), desc='Evaluating'):
        pred_path = args.pred_dir / f'{img_id}.png'
        if not pred_path.exists() and img_id in id_to_uuid:
            uuid = id_to_uuid[img_id]
            pred_path = args.pred_dir / f'{uuid}.png'
        if not pred_path.exists():
            missing_pred += 1
            continue
        gt_path = args.data_root / mask_file
        if not gt_path.exists():
            missing_gt += 1
            continue
        pred = load_binary_mask(pred_path, threshold=127)
        gt = load_binary_mask(gt_path, threshold=127)
        if pred.shape != gt.shape:
            pred_img = Image.fromarray((pred * 255).astype(np.uint8))
            pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
            pred = (np.array(pred_img) > 127).astype(np.uint8)
        metrics = compute_metrics(pred, gt)
        metrics['img_id'] = img_id
        all_metrics.append(metrics)
    if len(all_metrics) == 0:
        return 1
    mean_iou = np.mean([m['iou'] for m in all_metrics])
    mean_dice = np.mean([m['dice'] for m in all_metrics])
    mean_accuracy = np.mean([m['accuracy'] for m in all_metrics])
    mean_precision = np.mean([m['precision'] for m in all_metrics])
    mean_recall = np.mean([m['recall'] for m in all_metrics])
    results = {'fold': args.fold, 'num_evaluated': len(all_metrics), 'missing_predictions': missing_pred, 'missing_ground_truth': missing_gt, 'metrics': {'mean_iou': float(mean_iou), 'mean_dice': float(mean_dice), 'mean_accuracy': float(mean_accuracy), 'mean_precision': float(mean_precision), 'mean_recall': float(mean_recall)}, 'per_image_metrics': all_metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    if missing_pred > 0:
        pass
    if missing_gt > 0:
        pass
    return 0
if __name__ == '__main__':
    import sys
    sys.exit(main())
