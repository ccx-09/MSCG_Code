#!/usr/bin/env python3
"""
Usage:
    python datasets/ws1/modern_pipeline.py --input ../VOC2012 --output ../ws1_data/data/ws1 --debug
"""

import albumentations as A
from pathlib import Path
import json
import numpy as np
from PIL import Image
import argparse
from typing import Dict, List, Tuple
import logging
import uuid
import cv2
from sklearn.model_selection import StratifiedKFold, train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModernComplexityGrid:
    """
    Uses VOC2012 as source data for classical method compatibility.
    """

    def __init__(self, voc_root: str, output_dir: str):
        self.voc_root = Path(voc_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Industry-standard corruptions using Albumentations
        self.corruptions = {
            'clean': None,  # No corruption
            'gaussian_blur': A.GaussianBlur(blur_limit=(3,15), p=1.0),
            'shot_noise': A.ISONoise(color_shift=(0.01,0.1), intensity=(0.1,0.5), p=1.0),
            'fog': A.RandomFog(fog_coef_range=(0.3, 0.3), p=1.0)
        }
        
        # Scale bins (complexity levels) - computed from actual mask areas
        self.scale_bins = ['s1', 's2', 's3', 's4', 's5']  # <1%, 1-3%, 3-10%, 10-30%, >30%
        self.scale_thresholds = [0.01, 0.03, 0.10, 0.30, 1.0]  # Area ratio thresholds
        self.severity_levels = [0, 1, 2, 3, 4, 5]  # 0=clean, 1-5=increasing severity

    def _compute_scale_bin(self, mask_path: Path) -> str:
        """Compute scale bin based on actual object-to-image area ratio from VOC2012 mask"""
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return 's3'  # Default fallback

            object_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            area_ratio = object_pixels / total_pixels

            for i, threshold in enumerate(self.scale_thresholds):
                if area_ratio < threshold:
                    return self.scale_bins[i]
            return self.scale_bins[-1]  # Largest bin for >30%

        except Exception as e:
            logger.warning(f"Failed to compute scale bin for {mask_path}: {e}")
            return 's3'  # Default fallback

    def _get_corruption_with_severity(self, corruption_name: str, severity: int):
        """Get corruption transform scaled by severity level"""
        if corruption_name == 'clean' or severity == 0:
            return None

        if corruption_name == 'gaussian_blur':
            blur_min = 3 + severity * 2
            blur_max = 7 + severity * 4
            return A.GaussianBlur(blur_limit=(blur_min, blur_max), p=1.0)

        elif corruption_name == 'shot_noise':
            intensity_min = 0.05 + severity * 0.04
            intensity_max = 0.15 + severity * 0.04
            return A.ISONoise(
                color_shift=(0.01, 0.05 + severity * 0.01),
                intensity=(intensity_min, intensity_max),
                p=1.0
            )

        elif corruption_name == 'fog':
            fog_coef = 0.1 + severity * 0.14
            return A.RandomFog(
                fog_coef_range=(fog_coef, fog_coef + 0.1),
                p=1.0
            )

        return None  # Fallback

    def _assign_folds(self, sources: List[dict], n_splits: int = 5, seed: int = 42) -> List[List[dict]]:
        scale_to_int = {s: i for i, s in enumerate(self.scale_bins)}
        source_ids = np.array([s['voc_id'] for s in sources])
        scale_labels = np.array([scale_to_int[s['scale_bin']] for s in sources])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_sources = [[] for _ in range(n_splits)]
        for fold_idx, (_, test_idx) in enumerate(skf.split(source_ids, scale_labels)):
            fold_sources[fold_idx] = [sources[i] for i in test_idx]
        for i, fs in enumerate(fold_sources):
            logger.info(f"  Fold {i}: {len(fs)} sources")
        return fold_sources

    def _write_fold_json(self, records: List[dict], output_path: Path, fold: int, split: str):
        data = {
            'info': {'fold': fold, 'split': split, 'n_images': len(records)},
            'images': records
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_dataset(self, debug: bool = False):
        """
        Generate complexity grid dataset.
        """
        logger.info("Starting new VOC2012 MSCG pipeline")

        voc_images_dir = self.voc_root / "JPEGImages"
        voc_masks_dir = self.voc_root / "SegmentationClass"
        voc_train_split = self.voc_root / "ImageSets" / "Segmentation" / "train.txt"

        if not voc_images_dir.exists():
            logger.error(f"VOC2012 images not found: {voc_images_dir}")
            return
        if not voc_train_split.exists():
            logger.error(f"VOC2012 train split not found: {voc_train_split}")
            return

        with open(voc_train_split, 'r') as f:
            voc_ids = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Found {len(voc_ids)} VOC2012 training images")

        composites_dir = self.output_dir / "composites"
        masks_dir = self.output_dir / "masks"

        logger.info("Collecting valid sources with scale bins")
        valid_sources = []
        for voc_id in voc_ids:
            img_path = voc_images_dir / f"{voc_id}.jpg"
            msk_path = voc_masks_dir / f"{voc_id}.png"
            if not img_path.exists() or not msk_path.exists():
                continue
            try:
                msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
                if msk is None or np.sum(msk > 0) == 0:
                    continue
            except Exception:
                continue
            scale_bin = self._compute_scale_bin(msk_path)
            valid_sources.append({
                'voc_id': voc_id, 'image_path': img_path,
                'mask_path': msk_path, 'scale_bin': scale_bin
            })

        logger.info(f"Valid sources: {len(valid_sources)}")
        for sb in self.scale_bins:
            cnt = sum(1 for s in valid_sources if s['scale_bin'] == sb)
            logger.info(f"  {sb}: {cnt}")

        if debug:
            valid_sources = valid_sources[:20]
            logger.info(f"DEBUG: limiting to {len(valid_sources)} sources")

        logger.info("Assigning sources to folds via stratified k-fold")
        n_folds = 5
        fold_sources = self._assign_folds(valid_sources, n_splits=n_folds)

        combinations = []
        for corr in self.corruptions:
            for sev in self.severity_levels:
                combinations.append((corr, sev))
        logger.info(f"Generating {len(combinations)} combos per source ({len(valid_sources)} sources, total ~{len(valid_sources) * len(combinations)})")

        all_image_records = []
        for src in valid_sources:
            voc_id = src['voc_id']
            scale_bin = src['scale_bin']
            image = Image.open(src['image_path']).convert('RGB')
            w, h = image.width, image.height
            for corruption, severity in combinations:
                sample_uuid = str(uuid.uuid4())

                transform = self._get_corruption_with_severity(corruption, severity)
                if transform is None:
                    corrupted = np.array(image)
                else:
                    corrupted = transform(image=np.array(image))['image']

                composite_subdir = composites_dir / corruption / f"{scale_bin}_k{severity}"
                mask_subdir = masks_dir / f"{scale_bin}_k{severity}"
                composite_subdir.mkdir(parents=True, exist_ok=True)
                mask_subdir.mkdir(parents=True, exist_ok=True)

                jpg_name = f"{sample_uuid}.jpg"
                composite_path = composite_subdir / jpg_name
                Image.fromarray(corrupted).save(composite_path, quality=95)

                png_name = f"{sample_uuid}.png"
                mask_output_path = mask_subdir / png_name
                mask_img = cv2.imread(str(src['mask_path']))
                cv2.imwrite(str(mask_output_path), mask_img)

                file_name = f"{corruption}/{scale_bin}_k{severity}/{jpg_name}"
                mask_file = f"masks/{scale_bin}_k{severity}/{png_name}"
                all_image_records.append({
                    'id': len(all_image_records),
                    'file_name': file_name,
                    'mask_file': mask_file,
                    'corruption': corruption,
                    'scale_bin': scale_bin,
                    'severity': severity,
                    'source_voc_id': voc_id,
                    'width': w,
                    'height': h
                })

        logger.info(f"Total generated images: {len(all_image_records)}")

        folds_dir = self.output_dir / "folds"
        folds_dir.mkdir(exist_ok=True)

        voc_id_to_records = {}
        for rec in all_image_records:
            voc_id_to_records.setdefault(rec['source_voc_id'], []).append(rec)

        for fold_idx in range(n_folds):
            test_voc_ids = {s['voc_id'] for s in fold_sources[fold_idx]}
            test = []
            train_pool = []
            for voc_id, recs in voc_id_to_records.items():
                if voc_id in test_voc_ids:
                    test.extend(recs)
                else:
                    train_pool.extend(recs)

            train_pool_voc = list({r['source_voc_id'] for r in train_pool})
            train_voc, val_voc = train_test_split(
                train_pool_voc, test_size=0.15, random_state=42)
            train_voc_set = set(train_voc)
            val_voc_set = set(val_voc)

            train = [r for r in train_pool if r['source_voc_id'] in train_voc_set]
            val = [r for r in train_pool if r['source_voc_id'] in val_voc_set]

            for split_name, records in [('train', train), ('val', val), ('test', test)]:
                json_path = folds_dir / f"fold{fold_idx}_{split_name}.json"
                self._write_fold_json(records, json_path, fold_idx, split_name)

            logger.info(f"Fold {fold_idx}: train={len(train)}, val={len(val)}, test={len(test)}")

        logger.info(f"Output: {self.output_dir}")
        logger.info("Done")


def main():
    parser = argparse.ArgumentParser(
        description='MSCG Dataset Generation from VOC2012',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', default='../VOC2012',
                        help='VOC2012 dataset root directory')
    parser.add_argument('--output', default='../data/VOCdata',
                        help='Output directory')
    parser.add_argument('--debug', action='store_true',
                        help='Use only first 20 sources for testing')

    args = parser.parse_args()

    logger.info("MSCG Pipeline Starting")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    np.random.seed(42)

    generator = ModernComplexityGrid(args.input, args.output)
    generator.generate_dataset(debug=args.debug)

    logger.info("MSCG pipeline completed!")


if __name__ == "__main__":
    main()
