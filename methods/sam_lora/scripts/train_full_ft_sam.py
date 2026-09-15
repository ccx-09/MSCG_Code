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
from memory_utils import get_memory_info, optimize_memory_usage
from segment_anything import sam_model_registry
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

def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float=0.5) -> float:
    pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
    gt_binary = gt_mask.float()
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()

def create_full_ft_sam_model(sam_checkpoint: str, model_type: str='vit_h', device: str='cuda'):
    model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    model = model.to(device)
    for param in model.image_encoder.parameters():
        param.requires_grad = True
    for param in model.prompt_encoder.parameters():
        param.requires_grad = False
    for param in model.mask_decoder.parameters():
        param.requires_grad = True
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return model

def train_one_epoch(model, dataloader, optimizer, criterion_bce, criterion_dice, criterion_focal, scaler, device, epoch, gradient_accumulation_steps=128):
    model.train()
    total_loss = 0
    total_iou = 0
    num_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        with autocast():
            image_embeddings = model.image_encoder(images)
            B, _, H, W = images.shape
            prompt_points = torch.tensor([[H / 2, W / 2]], device=device).unsqueeze(0).repeat(B, 1, 1)
            prompt_labels = torch.ones((B, 1), device=device)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(points=(prompt_points, prompt_labels), boxes=None, masks=None)
            low_res_masks, _ = model.mask_decoder(image_embeddings=image_embeddings, image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False)
            pred_masks = F.interpolate(low_res_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss_bce = criterion_bce(pred_masks, masks)
            loss_dice = criterion_dice(pred_masks, masks)
            loss_focal = criterion_focal(pred_masks, masks)
            loss = loss_bce + loss_dice + loss_focal
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        with torch.no_grad():
            iou = compute_iou(pred_masks, masks)
            total_loss += loss.item() * gradient_accumulation_steps
            total_iou += iou
            num_batches += 1
        if (batch_idx + 1) % 10 == 0:
            pass
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    return (avg_loss, avg_iou)

def validate(model, dataloader, criterion_bce, criterion_dice, criterion_focal, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            image_embeddings = model.image_encoder(images)
            B, _, H, W = images.shape
            prompt_points = torch.tensor([[H / 2, W / 2]], device=device).unsqueeze(0).repeat(B, 1, 1)
            prompt_labels = torch.ones((B, 1), device=device)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(points=(prompt_points, prompt_labels), boxes=None, masks=None)
            low_res_masks, _ = model.mask_decoder(image_embeddings=image_embeddings, image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False)
            pred_masks = F.interpolate(low_res_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss_bce = criterion_bce(pred_masks, masks)
            loss_dice = criterion_dice(pred_masks, masks)
            loss_focal = criterion_focal(pred_masks, masks)
            loss = loss_bce + loss_dice + loss_focal
            iou = compute_iou(pred_masks, masks)
            total_loss += loss.item()
            total_iou += iou
            num_batches += 1
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    return (avg_loss, avg_iou)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--ws1_root', type=str, required=True)
    parser.add_argument('--sam_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sam_model', type=str, default='vit_h')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimize_memory_usage()
    model = create_full_ft_sam_model(sam_checkpoint=args.sam_checkpoint, model_type=args.sam_model, device=device)
    train_json = Path(args.ws1_root) / 'folds' / f'fold{args.fold}_train.json'
    val_json = Path(args.ws1_root) / 'folds' / f'fold{args.fold}_val.json'
    train_dataset = ModernSAMDataset(coco_json=str(train_json), data_root=str(args.ws1_root), prompt_policy='P1', img_size=1024, augment=True, split='train')
    val_dataset = ModernSAMDataset(coco_json=str(val_json), data_root=str(args.ws1_root), prompt_policy='P1', img_size=1024, augment=False, split='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    criterion_focal = FocalLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) // args.gradient_accumulation_steps * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scaler = GradScaler()
    best_val_iou = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion_bce, criterion_dice, criterion_focal, scaler, device, epoch, args.gradient_accumulation_steps)
        val_loss, val_iou = validate(model, val_loader, criterion_bce, criterion_dice, criterion_focal, device)
        scheduler.step()
        history.append({'epoch': epoch, 'train': {'loss': train_loss, 'iou': train_iou}, 'val': {'loss': val_loss, 'iou': val_iou}})
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_iou': best_val_iou, 'best_val_loss': val_loss, 'config': vars(args)}
            torch.save(checkpoint, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        if patience_counter >= args.early_stopping_patience:
            break
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
if __name__ == '__main__':
    main()
