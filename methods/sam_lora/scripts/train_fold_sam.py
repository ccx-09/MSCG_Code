#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from modern_dataset import ModernSAMDataset
from lora_inject import create_sam_lora_model, mark_only_lora_as_trainable
from memory_utils import get_memory_info, optimize_memory_usage
logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):

    def __init__(self, smooth: float=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):

    def __init__(self, alpha: float=1.0, gamma: float=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class SAMFoldTrainer:

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config['use_amp'] else None
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        self.resume_from_checkpoint = config.get('resume_from_checkpoint', None)

    def _make_collate_fn(self):
        policy = self.config.get('policy', 'P1')
        p_target = 2 if policy == 'P1' else 8

        def collate(batch):
            import torch as _torch
            images = _torch.stack([b['image'] for b in batch], dim=0)
            masks = _torch.stack([b['mask'] for b in batch], dim=0)
            pts_batch = []
            lbl_batch = []
            for b in batch:
                pts = b['prompts']['points']
                lbl = b['prompts']['labels']
                pts = pts.to(dtype=_torch.float32)
                lbl = lbl.to(dtype=_torch.long)
                pi = int(pts.shape[0]) if pts.ndim >= 2 else 0
                if pi >= p_target:
                    pts_fixed = pts[:p_target]
                    lbl_fixed = lbl[:p_target]
                else:
                    pts_fixed = _torch.zeros((p_target, 2), dtype=_torch.float32)
                    lbl_fixed = _torch.zeros((p_target,), dtype=_torch.long)
                    if pi > 0:
                        pts_fixed[:pi] = pts
                        lbl_fixed[:pi] = lbl
                pts_batch.append(pts_fixed)
                lbl_batch.append(lbl_fixed)
            points = _torch.stack(pts_batch, dim=0)
            labels = _torch.stack(lbl_batch, dim=0)
            return {'image': images, 'mask': masks, 'prompts': {'points': points, 'labels': labels}, 'meta': [b.get('meta', {}) for b in batch]}
        return collate

    def load_checkpoint_if_exists(self, resume_reason: str='manual'):
        checkpoint_path = None
        if self.resume_from_checkpoint:
            checkpoint_path = Path(self.resume_from_checkpoint)
        if not checkpoint_path or not checkpoint_path.exists():
            checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pt'))
            if checkpoints:
                checkpoint_path = checkpoints[-1]
        if checkpoint_path and checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.current_epoch = checkpoint['epoch'] + 1
                self.best_val_loss = checkpoint['best_val_loss']
                self.training_history = checkpoint.get('training_history', [])
                resume_event = {'epoch': self.current_epoch, 'event': 'resume', 'reason': resume_reason, 'checkpoint': str(checkpoint_path.name), 'resumed_from_epoch': checkpoint['epoch'], 'timestamp': datetime.now().isoformat()}
                self.training_history.append(resume_event)
                return True
            except Exception as e:
                return False
        return False

    def setup_datasets(self):
        fold = self.config['fold']
        ws1_root = Path(self.config['ws1_root'])
        manifest_dir = ws1_root / 'folds'
        train_json = manifest_dir / f'fold{fold}_train.json'
        val_json = manifest_dir / f'fold{fold}_val.json'
        if not train_json.exists() or not val_json.exists():
            raise FileNotFoundError(f'Fold manifests not found: {train_json}, {val_json}')
        self.train_dataset = ModernSAMDataset(coco_json=str(train_json), data_root=str(ws1_root), prompt_policy=self.config['policy'], img_size=self.config['img_size'], augment=True, split='train')
        self.val_dataset = ModernSAMDataset(coco_json=str(val_json), data_root=str(ws1_root), prompt_policy=self.config['policy'], img_size=self.config['img_size'], augment=False, split='val')
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'], pin_memory=True, drop_last=True, collate_fn=self._make_collate_fn())
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'], pin_memory=True, collate_fn=self._make_collate_fn())

    def setup_model(self):
        self.model = create_sam_lora_model(sam_checkpoint=self.config['sam_checkpoint'], model_type=self.config['sam_model'], lora_rank=self.config['lora_rank'], lora_alpha=self.config['lora_alpha'], lora_dropout=self.config['lora_dropout'], device=self.device)
        total_params = sum((p.numel() for p in self.model.parameters()))
        trainable_params = sum((p.numel() for p in self.model.parameters() if p.requires_grad))

    def setup_optimization(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        total_steps = len(self.train_loader) * self.config['epochs']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=self.config['lr'] * 0.1)

    def compute_loss(self, pred_masks: torch.Tensor, target_masks: torch.Tensor):
        dice_loss = self.dice_loss(pred_masks, target_masks)
        focal_loss = self.focal_loss(pred_masks, target_masks)
        total_loss = dice_loss + focal_loss
        return (total_loss, dice_loss, focal_loss)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        mark_only_lora_as_trainable(self.model)
        metrics = {'loss': 0.0, 'dice_loss': 0.0, 'focal_loss': 0.0}
        num_batches = len(self.train_loader)
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            prompts = {k: v.to(self.device) for k, v in batch['prompts'].items()}
            with autocast(enabled=self.config['use_amp']):
                imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                sam_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).div(255.0).view(1, 3, 1, 1)
                sam_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).div(255.0).view(1, 3, 1, 1)
                img01 = images * imagenet_std + imagenet_mean
                images_sam = (img01 - sam_mean) / sam_std
                image_embeddings = self.model.image_encoder(images_sam)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=(prompts['points'], prompts['labels']), boxes=None, masks=None)
                low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=image_embeddings, image_pe=self.model.prompt_encoder.get_dense_pe().expand(images.size(0), -1, -1, -1), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False)
                pred_masks = F.interpolate(low_res_masks, size=(self.config['img_size'], self.config['img_size']), mode='bilinear', align_corners=False)
                loss, dice_loss, focal_loss = self.compute_loss(pred_masks, masks)
                loss = loss / self.config['gradient_accumulation_steps']
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
            metrics['loss'] += loss.item() * self.config['gradient_accumulation_steps']
            metrics['dice_loss'] += dice_loss.item()
            metrics['focal_loss'] += focal_loss.item()
            if batch_idx % 50 == 0:
                pass
        for key in metrics:
            metrics[key] /= num_batches
        return metrics

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        metrics = {'loss': 0.0, 'dice_loss': 0.0, 'focal_loss': 0.0, 'iou': 0.0}
        num_batches = len(self.val_loader)
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                prompts = {k: v.to(self.device) for k, v in batch['prompts'].items()}
                with autocast(enabled=self.config['use_amp']):
                    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                    sam_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).div(255.0).view(1, 3, 1, 1)
                    sam_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).div(255.0).view(1, 3, 1, 1)
                    img01 = images * imagenet_std + imagenet_mean
                    images_sam = (img01 - sam_mean) / sam_std
                    image_embeddings = self.model.image_encoder(images_sam)
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=(prompts['points'], prompts['labels']), boxes=None, masks=None)
                    low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=image_embeddings, image_pe=self.model.prompt_encoder.get_dense_pe().expand(images.size(0), -1, -1, -1), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False)
                    pred_masks = F.interpolate(low_res_masks, size=(self.config['img_size'], self.config['img_size']), mode='bilinear', align_corners=False)
                    loss, dice_loss, focal_loss = self.compute_loss(pred_masks, masks)
                pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                intersection = (pred_binary * masks).sum()
                union = pred_binary.sum() + masks.sum() - intersection
                iou = intersection / (union + 1e-08)
                metrics['loss'] += loss.item()
                metrics['dice_loss'] += dice_loss.item()
                metrics['focal_loss'] += focal_loss.item()
                metrics['iou'] += iou.item()
        for key in metrics:
            metrics[key] /= num_batches
        return metrics

    def save_checkpoint(self, is_best: bool=False):

        def _get_adapter_state_dict() -> Dict[str, torch.Tensor]:
            full_sd = self.model.state_dict()
            adapter_sd = {}
            for k, v in full_sd.items():
                if '.lora_A.' in k or '.lora_B.' in k or k.startswith('mask_decoder.'):
                    adapter_sd[k] = v
            return adapter_sd
        save_adapter_only = bool(self.config.get('save_adapter_only', True))
        model_key = 'adapter_state_dict' if save_adapter_only else 'model_state_dict'
        checkpoint = {'epoch': self.current_epoch, 'fold': self.config['fold'], model_key: _get_adapter_state_dict() if save_adapter_only else self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(), 'best_val_loss': self.best_val_loss, 'config': self.config, 'sam_checkpoint': self.config.get('sam_checkpoint'), 'sam_model': self.config.get('sam_model', 'vit_h'), 'training_history': self.training_history}
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch:03d}.pt'
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
        keep_last_n = self.config.get('keep_last_n_checkpoints', 10)
        if keep_last_n and keep_last_n > 0:
            checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pt'))
            if len(checkpoints) > keep_last_n:
                to_delete = checkpoints[:-keep_last_n]
                for old_ckpt in to_delete:
                    old_ckpt.unlink()

    def preflight_checks(self):
        errors = []
        warnings = []
        manifest_dir = Path(self.config['ws1_root']) / 'folds'
        for split in ['train', 'val', 'test']:
            manifest = manifest_dir / f"fold{self.config['fold']}_{split}.json"
            if not manifest.exists():
                errors.append(f'Missing manifest: {manifest}')
        sam_ckpt = Path(self.config['sam_checkpoint'])
        if not sam_ckpt.exists():
            errors.append(f'SAM checkpoint not found: {sam_ckpt}')
        elif sam_ckpt.stat().st_size < 1000000000.0:
            warnings.append(f'SAM checkpoint suspiciously small: {sam_ckpt.stat().st_size / 1000000000.0:.2f}GB')
        if not torch.cuda.is_available():
            errors.append('CUDA not available - training will fail')
        else:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1000000000.0
            if gpu_mem < 10:
                warnings.append(f'GPU memory low: {gpu_mem:.1f}GB (recommend ≥12GB)')
        try:
            test_file = self.output_dir / '.write_test'
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            errors.append(f'Output directory not writable: {e}')
        import shutil
        stat = shutil.disk_usage(self.output_dir)
        free_gb = stat.free / 1000000000.0
        if free_gb < 10:
            warnings.append(f'Low disk space: {free_gb:.1f}GB free (recommend ≥20GB)')
        if warnings:
            for w in warnings:
                pass
        if errors:
            for e in errors:
                pass
            raise RuntimeError('Pre-flight validation failed - aborting before training')

    def train(self):
        self.preflight_checks()
        self.setup_datasets()
        self.setup_model()
        self.setup_optimization()
        resumed = self.load_checkpoint_if_exists()
        start_epoch = self.current_epoch
        if resumed:
            pass
        try:
            for epoch in range(start_epoch, self.config['epochs']):
                self.current_epoch = epoch
                try:
                    train_metrics = self.train_epoch()
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        self.save_checkpoint(is_best=False)
                        raise RuntimeError(f'Out of memory at epoch {epoch}. Reduce batch_size or gradient_accumulation_steps.') from e
                    raise
                val_metrics = self.validate()
                if np.isnan(train_metrics['loss']) or np.isnan(val_metrics['loss']):
                    self.save_checkpoint(is_best=False)
                    raise RuntimeError(f'Training diverged (NaN loss) at epoch {epoch}. Try reducing learning rate.')
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                history_entry = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics}
                self.training_history.append(history_entry)
                self.save_checkpoint(is_best=False)
                if self.patience_counter >= self.config['early_stopping_patience']:
                    break
        except KeyboardInterrupt:
            self.save_checkpoint(is_best=False)
            raise
        except Exception as e:
            try:
                self.save_checkpoint(is_best=False)
            except Exception as save_err:
                pass
            raise
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--ws1_root', required=True)
    parser.add_argument('--sam_checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--policy', choices=['P1', 'P2'], default='P1')
    parser.add_argument('--sam_model', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_h')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--keep_last_n_checkpoints', type=int, default=10)
    parser.add_argument('--save_adapter_only', type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)
    parser.add_argument('--save_full_model', action='store_true', default=False)
    parser.add_argument('--resume_from_checkpoint')
    args = parser.parse_args()
    save_adapter_only = args.save_adapter_only
    if args.save_full_model:
        save_adapter_only = False
    config = {'fold': args.fold, 'ws1_root': args.ws1_root, 'sam_checkpoint': args.sam_checkpoint, 'output_dir': args.output_dir, 'policy': args.policy, 'sam_model': args.sam_model, 'lora_rank': args.lora_rank, 'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout, 'img_size': 1024, 'batch_size': args.batch_size, 'gradient_accumulation_steps': args.gradient_accumulation_steps, 'epochs': args.epochs, 'lr': args.lr, 'weight_decay': args.weight_decay, 'warmup_ratio': args.warmup_ratio, 'use_amp': True, 'max_grad_norm': 1.0, 'early_stopping_patience': args.early_stopping_patience, 'num_workers': args.num_workers, 'seed': args.seed, 'keep_last_n_checkpoints': args.keep_last_n_checkpoints, 'save_adapter_only': save_adapter_only, 'resume_from_checkpoint': args.resume_from_checkpoint}
    trainer = SAMFoldTrainer(config)
    trainer.train()
if __name__ == '__main__':
    main()
