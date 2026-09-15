#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pickle
import xgboost as xgb
from pathlib import Path
import cv2
from PIL import Image
import logging
from tqdm import tqdm
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import unary_from_softmax
logger = logging.getLogger(__name__)
CRF_PARAMS = {'bilateral_sxy': (80, 80), 'bilateral_srgb': (13, 13, 13), 'bilateral_compat': 10, 'gaussian_sxy': (3, 3), 'gaussian_compat': 3, 'n_steps': 5}

def load_model_and_preprocessor(model_dir: Path, fold: int):
    model_path = model_dir / f'fold{fold}_model.json'
    preprocessor_path = model_dir / f'fold{fold}_preprocessor.pkl'
    model = xgb.Booster()
    model.load_model(str(model_path))
    with open(preprocessor_path, 'rb') as f:
        preprocessor_data = pickle.load(f)
    preprocessor = preprocessor_data['pipeline']
    return (model, preprocessor)

def load_test_features(features_dir: Path, fold: int):
    test_path = features_dir / f'fold{fold}_test_features.npz'
    test_data = np.load(test_path, allow_pickle=True)
    X_test = test_data['features']
    y_test = test_data['labels']
    image_indices = test_data['image_indices']
    patch_coords = test_data['patch_coords']
    uuid_list = test_data['uuid_list']
    n_images = test_data['n_images'].item()
    patch_size = test_data['patch_size'].item()
    stride = test_data['stride'].item()
    return (X_test, y_test, image_indices, patch_coords, uuid_list, n_images, patch_size, stride)

def reconstruct_probability_map(probabilities, image_indices, patch_coords, image_shape, patch_size, image_idx):
    h, w = image_shape[:2]
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    mask = image_indices == image_idx
    img_probs = probabilities[mask]
    img_coords = patch_coords[mask]
    for prob, (x, y) in zip(img_probs, img_coords):
        prob_map[y:y + patch_size, x:x + patch_size] += prob
        count_map[y:y + patch_size, x:x + patch_size] += 1
    prob_map = np.divide(prob_map, count_map, where=count_map > 0)
    return prob_map

def apply_crf(prob_map, image, params=CRF_PARAMS):
    h, w = prob_map.shape
    n_labels = 2
    crf = dcrf.DenseCRF2D(w, h, n_labels)
    prob_stack = np.stack([1 - prob_map, prob_map], axis=0)
    unary = unary_from_softmax(prob_stack)
    crf.setUnaryEnergy(unary)
    crf.addPairwiseGaussian(sxy=params['gaussian_sxy'], compat=params['gaussian_compat'], kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    crf.addPairwiseBilateral(sxy=params['bilateral_sxy'], srgb=params['bilateral_srgb'], rgbim=image, compat=params['bilateral_compat'], kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = crf.inference(params['n_steps'])
    Q = np.array(Q).reshape((n_labels, h, w))
    return Q[1]

def generate_predictions(model, preprocessor, X_test, image_indices, patch_coords, uuid_list, fold_manifest, ws1_root, output_dir, patch_size):
    with open(fold_manifest, 'r') as f:
        manifest = json.load(f)
    uuid_to_info = {}
    for img_info in manifest['images']:
        img_uuid = Path(img_info['file_name']).stem
        uuid_to_info[img_uuid] = img_info
    X_test_processed = preprocessor.transform(X_test)
    dtest = xgb.DMatrix(X_test_processed)
    probabilities = model.predict(dtest)
    predictions_dir = Path(output_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    ws1_root = Path(ws1_root)
    composites_dir = ws1_root / 'composites'
    unique_uuids = []
    seen = set()
    for uuid in uuid_list:
        if uuid not in seen:
            unique_uuids.append(uuid)
            seen.add(uuid)
    for img_idx, img_uuid in enumerate(tqdm(unique_uuids, desc='Generating predictions')):
        if img_uuid not in uuid_to_info:
            continue
        img_info = uuid_to_info[img_uuid]
        img_path = composites_dir / img_info['file_name']
        try:
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            continue
        prob_map = reconstruct_probability_map(probabilities, image_indices, patch_coords, image.shape, patch_size, img_idx)
        prob_map_crf = prob_map
        pred_mask = (prob_map_crf > 0.5).astype(np.uint8) * 255
        pred_path = predictions_dir / f'{img_uuid}.png'
        cv2.imwrite(str(pred_path), pred_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--features_dir', required=True)
    parser.add_argument('--fold_manifest', required=True)
    parser.add_argument('--ws1_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--fold', type=int, required=True)
    args = parser.parse_args()
    for key, val in CRF_PARAMS.items():
        pass
    model_dir = Path(args.model_dir)
    model, preprocessor = load_model_and_preprocessor(model_dir, args.fold)
    features_dir = Path(args.features_dir)
    X_test, y_test, image_indices, patch_coords, uuid_list, n_images, patch_size, stride = load_test_features(features_dir, args.fold)
    generate_predictions(model, preprocessor, X_test, image_indices, patch_coords, uuid_list, args.fold_manifest, args.ws1_root, args.output_dir, patch_size)
    output_dir = Path(args.output_dir)
    pred_files = list(output_dir.glob('*.png'))
    if len(pred_files) != n_images:
        pass
if __name__ == '__main__':
    main()
