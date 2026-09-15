#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pickle
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.pipeline import Pipeline
import logging
import time
import yaml
logger = logging.getLogger(__name__)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def load_features(features_dir: Path, fold: int):
    train_path = features_dir / f'fold{fold}_train_features.npz'
    val_path = features_dir / f'fold{fold}_val_features.npz'
    train_data = np.load(train_path, allow_pickle=True)
    X_train = train_data['features']
    y_train = train_data['labels']
    train_uuids = train_data['uuid_list']
    val_data = np.load(val_path, allow_pickle=True)
    X_val = val_data['features']
    y_val = val_data['labels']
    val_uuids = val_data['uuid_list']
    return (X_train, y_train, X_val, y_val, train_uuids, val_uuids)

def create_preprocessor(X_train, y_train, k_features=256):
    preprocessor = Pipeline([('scaler', StandardScaler()), ('selector', SelectKBest(score_func=f_classif, k=min(k_features, X_train.shape[1])))])
    preprocessor.fit(X_train, y_train)
    selected_indices = preprocessor.named_steps['selector'].get_support(indices=True)
    return (preprocessor, selected_indices)

def compute_sample_weights(y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.zeros(len(y_train))
    for idx, cls in enumerate(classes):
        sample_weights[y_train == cls] = class_weights[idx]
    return sample_weights

def train_xgboost(X_train, y_train, X_val, y_val, sample_weights, config):
    hyperparams = config['model']['hyperparameters']
    base_params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'gpu_hist' if CUPY_AVAILABLE else 'hist', 'random_state': 42, 'verbosity': 1}
    best_model = None
    best_score = -float('inf')
    best_params = None
    params = base_params.copy()
    params.update({'max_depth': hyperparams['max_depth'][0], 'learning_rate': hyperparams['learning_rate'][0], 'n_estimators': hyperparams['n_estimators'][0], 'subsample': hyperparams['subsample'][0], 'colsample_bytree': hyperparams['colsample_bytree'][0], 'reg_alpha': hyperparams['reg_alpha'][0], 'reg_lambda': hyperparams['reg_lambda'][0]})
    if CUPY_AVAILABLE:
        try:
            X_train_gpu = cp.asarray(X_train)
            y_train_gpu = cp.asarray(y_train)
            X_val_gpu = cp.asarray(X_val)
            y_val_gpu = cp.asarray(y_val)
            weights_gpu = cp.asarray(sample_weights)
            dtrain = xgb.DeviceQuantileDMatrix(X_train_gpu, label=y_train_gpu, weight=weights_gpu)
            dval = xgb.DeviceQuantileDMatrix(X_val_gpu, label=y_val_gpu)
        except Exception as e:
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
            dval = xgb.DMatrix(X_val, label=y_val)
    else:
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train, weight=sample_weights)
        dval = xgb.QuantileDMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dval, 'val')]
    n_estimators = params.pop('n_estimators')
    model = xgb.train(params, dtrain, num_boost_round=n_estimators, evals=evals, early_stopping_rounds=50, verbose_eval=10)
    y_pred = model.predict(dval)
    y_pred_binary = (y_pred > 0.5).astype(int)
    val_acc = accuracy_score(y_val, y_pred_binary)
    val_f1 = f1_score(y_val, y_pred_binary)
    val_iou = jaccard_score(y_val, y_pred_binary)
    return (model, params, {'accuracy': val_acc, 'f1': val_f1, 'iou': val_iou})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--k_features', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(args.features_dir)
    X_train, y_train, X_val, y_val, train_uuids, val_uuids = load_features(features_dir, args.fold)
    preprocessor, selected_indices = create_preprocessor(X_train, y_train, args.k_features)
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    sample_weights = compute_sample_weights(y_train)
    start_time = time.time()
    model, params, metrics = train_xgboost(X_train_processed, y_train, X_val_processed, y_val, sample_weights, config)
    train_time = time.time() - start_time
    model_path = output_dir / f'fold{args.fold}_model.json'
    model.save_model(str(model_path))
    preprocessor_path = output_dir / f'fold{args.fold}_preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({'pipeline': preprocessor, 'selected_indices': selected_indices, 'n_features_in': X_train.shape[1], 'n_features_out': X_train_processed.shape[1], 'fold': args.fold, 'seed': args.seed}, f)
    summary = {'fold': args.fold, 'train_samples': len(y_train), 'val_samples': len(y_val), 'n_features_original': X_train.shape[1], 'n_features_selected': X_train_processed.shape[1], 'model_params': params, 'validation_metrics': metrics, 'training_time': train_time, 'class_distribution': {'train': {'0': int(np.sum(y_train == 0)), '1': int(np.sum(y_train == 1))}, 'val': {'0': int(np.sum(y_val == 0)), '1': int(np.sum(y_val == 1))}}}
    summary_path = output_dir / f'fold{args.fold}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    if metrics['iou'] < 0.3:
        pass
if __name__ == '__main__':
    main()
