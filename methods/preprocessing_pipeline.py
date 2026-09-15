#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import argparse
logger = logging.getLogger(__name__)

class ClassicalPreprocessor:

    def __init__(self, scaler_type: str='standard', feature_selection: Optional[str]=None, n_features: Optional[int]=None, test_size: float=0.2, val_size: float=0.2, random_state: int=42):
        self.scaler_type = scaler_type
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = None
        self.feature_selector = None
        self.class_weights = None
        self.feature_names = None
        self.selected_feature_names = None
        self.stats = {}

    def fit(self, features: np.ndarray, labels: np.ndarray, feature_names: List[str]) -> 'ClassicalPreprocessor':
        self.feature_names = feature_names.copy()
        assert features.shape[0] == len(labels), f'Feature-label mismatch: {features.shape[0]} vs {len(labels)}'
        assert features.shape[1] == len(feature_names), f'Feature-name mismatch: {features.shape[1]} vs {len(feature_names)}'
        finite_mask = np.isfinite(features)
        finite_ratio = np.mean(finite_mask)
        if finite_ratio < 0.99:
            for i in range(features.shape[1]):
                col_mask = finite_mask[:, i]
                if np.any(~col_mask):
                    median_val = np.median(features[col_mask, i])
                    features[~col_mask, i] = median_val
        unique_labels = np.unique(labels)
        self.class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
        self.class_weight_dict = dict(zip(unique_labels, self.class_weights))
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'none':
            self.scaler = None
        else:
            raise ValueError(f'Unknown scaler type: {self.scaler_type}')
        if self.scaler:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = features.copy()
        if self.feature_selection == 'k_best':
            n_features = self.n_features or min(20, features_scaled.shape[1])
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            features_selected = self.feature_selector.fit_transform(features_scaled, labels)
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_feature_names = [feature_names[i] for i in selected_indices]
        elif self.feature_selection == 'mutual_info':
            n_features = self.n_features or min(20, features_scaled.shape[1])
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            features_selected = self.feature_selector.fit_transform(features_scaled, labels)
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_feature_names = [feature_names[i] for i in selected_indices]
        elif self.feature_selection is None:
            features_selected = features_scaled
            self.selected_feature_names = feature_names.copy()
        else:
            raise ValueError(f'Unknown feature selection method: {self.feature_selection}')
        self.stats = {'n_samples': features.shape[0], 'n_original_features': features.shape[1], 'n_selected_features': features_selected.shape[1], 'class_distribution': dict(zip(*np.unique(labels, return_counts=True))), 'class_weights': self.class_weight_dict, 'finite_values_ratio': float(finite_ratio), 'feature_stats': {'original': {'mean': features.mean(axis=0).tolist(), 'std': features.std(axis=0).tolist(), 'min': features.min(axis=0).tolist(), 'max': features.max(axis=0).tolist()}, 'processed': {'mean': features_selected.mean(axis=0).tolist(), 'std': features_selected.std(axis=0).tolist(), 'min': features_selected.min(axis=0).tolist(), 'max': features_selected.max(axis=0).tolist()}}}
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.feature_names is None:
            raise ValueError('Preprocessor not fitted yet. Call fit() first.')
        finite_mask = np.isfinite(features)
        for i in range(features.shape[1]):
            col_mask = finite_mask[:, i]
            if np.any(~col_mask):
                mean_val = self.stats['feature_stats']['original']['mean'][i]
                features[~col_mask, i] = mean_val
        if self.scaler:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features.copy()
        if self.feature_selector:
            features_selected = self.feature_selector.transform(features_scaled)
        else:
            features_selected = features_scaled
        return features_selected

    def fit_transform(self, features: np.ndarray, labels: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(features, labels, feature_names)
        return (self.transform(features), labels)

    def create_splits(self, features: np.ndarray, labels: np.ndarray, metadata: Optional[List[Dict]]=None) -> Dict[str, Dict[str, Union[np.ndarray, List]]]:
        if metadata is not None:
            X_trainval, X_test, y_trainval, y_test, meta_trainval, meta_test = train_test_split(features, labels, metadata, test_size=self.test_size, stratify=labels, random_state=self.random_state)
        else:
            X_trainval, X_test, y_trainval, y_test = train_test_split(features, labels, test_size=self.test_size, stratify=labels, random_state=self.random_state)
            meta_trainval = meta_test = None
        val_size_adjusted = self.val_size / (1 - self.test_size)
        if metadata is not None:
            X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(X_trainval, y_trainval, meta_trainval, test_size=val_size_adjusted, stratify=y_trainval, random_state=self.random_state)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_size_adjusted, stratify=y_trainval, random_state=self.random_state)
            meta_train = meta_val = None
        splits = {'train': {'features': X_train, 'labels': y_train, 'metadata': meta_train}, 'val': {'features': X_val, 'labels': y_val, 'metadata': meta_val}, 'test': {'features': X_test, 'labels': y_test, 'metadata': meta_test}}
        for split_name, split_data in splits.items():
            labels = split_data['labels']
            class_dist = dict(zip(*np.unique(labels, return_counts=True)))
        return splits

    def save(self, filepath: str) -> None:
        save_data = {'scaler': self.scaler, 'feature_selector': self.feature_selector, 'class_weights': self.class_weights, 'class_weight_dict': self.class_weight_dict, 'feature_names': self.feature_names, 'selected_feature_names': self.selected_feature_names, 'stats': self.stats, 'config': {'scaler_type': self.scaler_type, 'feature_selection': self.feature_selection, 'n_features': self.n_features, 'test_size': self.test_size, 'val_size': self.val_size, 'random_state': self.random_state}}
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, filepath: str) -> 'ClassicalPreprocessor':
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        config = save_data['config']
        preprocessor = cls(**config)
        preprocessor.scaler = save_data['scaler']
        preprocessor.feature_selector = save_data['feature_selector']
        preprocessor.class_weights = save_data['class_weights']
        preprocessor.class_weight_dict = save_data['class_weight_dict']
        preprocessor.feature_names = save_data['feature_names']
        preprocessor.selected_feature_names = save_data['selected_feature_names']
        preprocessor.stats = save_data['stats']
        return preprocessor

def build_preprocessing_pipeline(features_path: str, output_dir: str, config: Dict) -> None:
    data = np.load(features_path, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    feature_names = data['feature_names'].tolist()
    metadata = data['metadata'].tolist() if 'metadata' in data else None
    if metadata is not None and len(metadata) != int(features.shape[0]):
        metadata = None
    preprocessor = ClassicalPreprocessor(**config)
    features_processed, labels_processed = preprocessor.fit_transform(features, labels, feature_names)
    splits = preprocessor.create_splits(features_processed, labels_processed, metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(output_dir / 'preprocessor.pkl')
    for split_name, split_data in splits.items():
        split_file = output_dir / f'{split_name}_split.npz'
        np.savez_compressed(split_file, features=split_data['features'], labels=split_data['labels'], metadata=split_data['metadata'], feature_names=preprocessor.selected_feature_names)
    config_stats = {'preprocessing_config': config, 'pipeline_stats': preprocessor.stats, 'selected_features': preprocessor.selected_feature_names}
    config_file = output_dir / 'preprocessing_stats.json'
    with open(config_file, 'w') as f:
        json.dump(config_stats, f, indent=2)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scaler', choices=['standard', 'robust', 'none'], default='standard')
    parser.add_argument('--feature_selection', choices=['k_best', 'mutual_info'])
    parser.add_argument('--n_features', type=int)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    config = {'scaler_type': args.scaler, 'feature_selection': args.feature_selection, 'n_features': args.n_features, 'test_size': args.test_size, 'val_size': args.val_size, 'random_state': args.random_state}
    build_preprocessing_pipeline(args.features, args.output, config)
