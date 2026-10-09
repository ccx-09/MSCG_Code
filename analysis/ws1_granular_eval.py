#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import csv

def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im, dtype=np.uint8)

def compute_confusion(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int=255) -> np.ndarray:
    mask = gt != ignore_index
    pred_v = pred[mask].astype(np.int64)
    gt_v = gt[mask].astype(np.int64)
    cm = np.bincount(gt_v * num_classes + pred_v, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)

def iou_dice_acc_from_confusion(cm: np.ndarray) -> Tuple[float, float, float]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(denom > 0, tp / denom, np.nan)
    miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
    total = cm.sum()
    correct = tp.sum()
    acc = float(correct / total) if total > 0 else 0.0
    denom_d = 2 * tp + fp + fn
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = np.where(denom_d > 0, 2 * tp / denom_d, np.nan)
    mdice = float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0
    return (miou, mdice, acc)

def brittleness_coefficient(per_cell: Dict[Tuple[str, str, str], Dict[str, float]], scales_order: List[str], sevs_order: List[str]) -> float:
    path = []
    n = min(len(scales_order), len(sevs_order))
    for i in range(n):
        scale = scales_order[-(i + 1)]
        sev = sevs_order[i]
        key = (scale, sev, 'clean')
        if key in per_cell:
            path.append((i, per_cell[key]['miou']))
    if len(path) < 2:
        return float('nan')
    xs = np.array([p[0] for p in path], dtype=np.float64)
    ys = np.array([p[1] for p in path], dtype=np.float64)
    A = np.vstack([xs, np.ones_like(xs)]).T
    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(-m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ws1-json', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--gt-dir', required=True)
    ap.add_argument('--num-classes', type=int, default=2)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    ws1_json = Path(args.ws1_json)
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    try:
        ws1_root = gt_dir.parents[0]
    except IndexError:
        ws1_root = gt_dir.parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(ws1_json.read_text())
    images = data.get('images', [])
    meta = {}
    scales_seen, sevs_seen, corrs_seen = (set(), set(), set())
    for img in images:
        file_name = str(img['file_name']).replace('\\', '/').lstrip('./')
        uuid = Path(file_name).stem
        m = {'scale_bin': img.get('scale_bin'), 'severity': str(img.get('severity')), 'corruption': img.get('corruption')}
        meta[uuid] = m
        scales_seen.add(m['scale_bin'])
        sevs_seen.add(m['severity'])
        corrs_seen.add(m['corruption'])
    scales_order = sorted(scales_seen, key=lambda s: int(s[1:]) if s and len(s) > 1 and s[1:].isdigit() else s)
    sevs_order = sorted(sevs_seen, key=lambda x: int(x))
    corr_order = sorted(corrs_seen)
    overall_cm = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    cell_cm: Dict[Tuple[str, str, str], np.ndarray] = {}
    scale_cm: Dict[str, np.ndarray] = {}
    per_image_rows = []
    pred_files = sorted(pred_dir.rglob('*.png'))
    strip_suffixes = ['_crf', '_raw', '_pred', '_mask', '_seg', '-crf', '-raw']
    matched = 0
    skipped_no_meta = 0
    skipped_no_gt = 0
    for pred_path in pred_files:
        stem = pred_path.stem
        uuid = stem
        for suf in strip_suffixes:
            if uuid.endswith(suf):
                uuid = uuid[:-len(suf)]
                break
        m = meta.get(uuid)
        if not m:
            skipped_no_meta += 1
            continue
        scale = m['scale_bin']
        sev = m['severity']
        gt_path = None
        for rec in images:
            fn = str(rec.get('file_name', '')).replace('\\', '/').lstrip('./')
            if Path(fn).stem == uuid:
                mf = rec.get('mask_file')
                if mf:
                    gt_path = (ws1_root / str(mf).replace('\\', '/').lstrip('./')).resolve()
                break
        if gt_path is None:
            gt_path = gt_dir / f'{scale}_k{sev}' / f'{uuid}.png'
        if not gt_path.exists():
            alt = gt_dir / f'{uuid}.png'
            if alt.exists():
                gt_path = alt
            else:
                skipped_no_gt += 1
                continue
        pred = load_mask(pred_path)
        gt = load_mask(gt_path)
        if gt.ndim == 3:
            if args.num_classes == 2:
                gt = (gt.sum(axis=2) > 0).astype(np.uint8)
            else:
                gt = gt[:, :, 0].astype(np.uint8)
        if pred.ndim == 3:
            if args.num_classes == 2:
                pred = (pred.sum(axis=2) > 0).astype(np.uint8)
            else:
                pred = pred[:, :, 0].astype(np.uint8)
        if pred.shape != gt.shape:
            from PIL import Image as _PILImage
            pred = np.array(_PILImage.fromarray(pred, mode='L').resize((gt.shape[1], gt.shape[0]), resample=_PILImage.NEAREST), dtype=np.uint8)
        if args.num_classes == 2:
            gt = (gt > 0).astype(np.uint8)
            pred = (pred > 0).astype(np.uint8)
        cm = compute_confusion(pred, gt, args.num_classes)
        overall_cm += cm
        if scale not in scale_cm:
            scale_cm[scale] = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
        scale_cm[scale] += cm
        matched += 1
        key = (m['scale_bin'], m['severity'], m['corruption'])
        if key not in cell_cm:
            cell_cm[key] = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
        cell_cm[key] += cm
        miou_i, mdice_i, acc_i = iou_dice_acc_from_confusion(cm)
        per_image_rows.append({'uuid': uuid, 'corruption': m['corruption'], 'scale_bin': m['scale_bin'], 'severity': m['severity'], 'miou': f'{miou_i:.6f}', 'dice': f'{mdice_i:.6f}', 'accuracy': f'{acc_i:.6f}'})
    per_image_csv = out_dir / 'per_image.csv'
    with open(per_image_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['uuid', 'corruption', 'scale_bin', 'severity', 'miou', 'dice', 'accuracy'])
        writer.writeheader()
        writer.writerows(per_image_rows)
    per_cell = {}
    per_cell_csv = out_dir / 'per_cell.csv'
    with open(per_cell_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['corruption', 'scale_bin', 'severity', 'count', 'miou', 'dice', 'accuracy'])
        writer.writeheader()
        for corr in corr_order:
            for scale in scales_order:
                for sev in sevs_order:
                    key = (scale, sev, corr)
                    cm = cell_cm.get(key)
                    if cm is None:
                        continue
                    miou_c, mdice_c, acc_c = iou_dice_acc_from_confusion(cm)
                    count = int(cm.sum())
                    per_cell[key] = {'miou': miou_c, 'dice': mdice_c, 'accuracy': acc_c}
                    writer.writerow({'corruption': corr, 'scale_bin': scale, 'severity': sev, 'count': count, 'miou': f'{miou_c:.6f}', 'dice': f'{mdice_c:.6f}', 'accuracy': f'{acc_c:.6f}'})

    # Audit artifact for the scale-level aggregation.  Each row is the
    # fold-level confusion matrix pooled over all corruptions and severities
    # in that scale bin; it is deliberately kept separate from per_cell.csv.
    per_scale_csv = out_dir / 'per_scale.csv'
    with open(per_scale_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['scale_bin', 'tn', 'fp', 'fn', 'tp', 'pixels', 'pooled_miou', 'pooled_dice', 'pooled_accuracy', 'cell_macro_miou'])
        writer.writeheader()
        for scale in scales_order:
            cm = scale_cm.get(scale)
            if cm is None:
                continue
            if args.num_classes != 2:
                raise ValueError('per_scale.csv audit output requires binary evaluation (num_classes=2)')
            miou_s, mdice_s, acc_s = iou_dice_acc_from_confusion(cm)
            cell_vals = [per_cell[scale, sev, corr]['miou'] for sev in sevs_order for corr in corr_order if (scale, sev, corr) in per_cell]
            writer.writerow({
                'scale_bin': scale,
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1]),
                'pixels': int(cm.sum()),
                'pooled_miou': f'{miou_s:.6f}',
                'pooled_dice': f'{mdice_s:.6f}',
                'pooled_accuracy': f'{acc_s:.6f}',
                'cell_macro_miou': f'{float(np.mean(cell_vals)):.6f}' if cell_vals else 'nan',
            })

    overall_miou, overall_mdice, overall_acc = iou_dice_acc_from_confusion(overall_cm)

    def agg_by_scale_pooled():
        rows = []
        for scale in scales_order:
            cm = scale_cm.get(scale)
            if cm is None:
                rows.append((scale, float('nan')))
            else:
                miou_s, _, _ = iou_dice_acc_from_confusion(cm)
                rows.append((scale, miou_s))
        return rows

    def agg_by_scale_cell_macro():
        rows = []
        for scale in scales_order:
            vals = [per_cell[scale, sev, corr]['miou'] for sev in sevs_order for corr in corr_order if (scale, sev, corr) in per_cell]
            rows.append((scale, float(np.mean(vals)) if vals else float('nan')))
        return rows

    def agg_by_sev():
        rows = []
        for sev in sevs_order:
            vals = [per_cell[scale, sev, corr]['miou'] for scale in scales_order for corr in corr_order if (scale, sev, corr) in per_cell]
            rows.append((sev, float(np.mean(vals)) if vals else float('nan')))
        return rows

    def agg_by_corr():
        rows = []
        for corr in corr_order:
            vals = [per_cell[scale, sev, corr]['miou'] for scale in scales_order for sev in sevs_order if (scale, sev, corr) in per_cell]
            rows.append((corr, float(np.mean(vals)) if vals else float('nan')))
        return rows
    bc = brittleness_coefficient(per_cell, scales_order, sevs_order)
    scale_cell_macro = dict(agg_by_scale_cell_macro())
    scale_pooled = dict(agg_by_scale_pooled())
    summary = {'overall': {'miou': overall_miou, 'dice': overall_mdice, 'accuracy': overall_acc}, 'per_scale_miou': scale_cell_macro, 'per_scale_cell_macro_miou': scale_cell_macro, 'per_scale_pooled_miou': scale_pooled, 'per_severity_miou': dict(agg_by_sev()), 'per_corruption_miou': dict(agg_by_corr()), 'brittleness_coefficient': bc, 'meta': {'num_classes': args.num_classes, 'ws1_json': str(ws1_json), 'pred_dir': str(pred_dir), 'gt_dir': str(gt_dir), 'num_pred_files_found': len(pred_files), 'num_images_in_json': len(images), 'num_pred_matched': matched, 'num_skipped_no_meta': skipped_no_meta, 'num_skipped_no_gt': skipped_no_gt}}
    out_json = out_dir / 'summary.json'
    out_json.write_text(json.dumps(summary, indent=2))
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        for corr in corr_order:
            rows = []
            for scale in scales_order:
                for sev in sevs_order:
                    key = (scale, sev, corr)
                    v = per_cell.get(key, {}).get('miou', np.nan)
                    rows.append({'scale_bin': scale, 'severity': sev, 'miou': v})
            df = pd.DataFrame(rows)
            pivot = df.pivot(index='scale_bin', columns='severity', values='miou').reindex(index=scales_order, columns=sevs_order)
            plt.figure(figsize=(8, 3.5))
            im = plt.imshow(pivot.values, cmap='viridis', vmin=0.0, vmax=1.0)
            plt.title(f'WS1 mIoU: {corr}')
            plt.xticks(range(len(sevs_order)), [f'k{s}' for s in sevs_order])
            plt.yticks(range(len(scales_order)), scales_order)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.values[i, j]
                    if np.isfinite(v):
                        plt.text(j, i, f'{v:.2f}', ha='center', va='center', color='white', fontsize=7)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            out_png = out_dir / f'heatmap_miou_{corr}.png'
            plt.savefig(out_png, dpi=150)
            plt.close()
    except Exception as e:
        pass
    return 0
if __name__ == '__main__':
    import sys
    sys.exit(main())
