#!/usr/bin/env python3
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

class WS1Dataset(Dataset):

    def __init__(self, json_path, data_root, transform=None):
        with open(json_path) as f:
            data = json.load(f)
        self.images = data['images']
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = self.data_root / 'composites' / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        mask_file = img_info.get('mask_file')
        if mask_file:
            mask_path = self.data_root / mask_file
            if not mask_path.exists():
                raise FileNotFoundError(f'Mask not found: {mask_path}')
            mask = Image.open(mask_path).convert('L')
        else:
            raise ValueError(f'No mask_file found for image {img_id}')
        image = image.resize((512, 512), Image.BILINEAR)
        mask = mask.resize((512, 512), Image.NEAREST)
        image_np = np.array(image) / 255.0
        mask_np = np.array(mask)
        mask_binary = (mask_np > 0).astype(np.float32)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()
        return (image_tensor, mask_tensor, img_id)

def compute_iou(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = (intersection + 1e-07) / (union + 1e-07)
    return iou.item()

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_iou = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, masks, img_ids) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        if batch_idx == 0 and epoch == 1:
            pass
        optimizer.zero_grad()
        logits = model(images)
        if batch_idx == 0 and epoch == 1:
            pass
        loss = criterion(logits, masks)
        if batch_idx == 0 and epoch == 1:
            pass
        loss.backward()
        if batch_idx == 0 and epoch == 1:
            has_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 0:
                        has_grad = True
                        break
            if not has_grad:
                pass
        optimizer.step()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            iou = compute_iou(probs, masks)
        total_loss += loss.item()
        total_iou += iou
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    return (avg_loss, avg_iou)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for images, masks, img_ids in tqdm(dataloader, desc='[Val]'):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits)
            iou = compute_iou(probs, masks)
            total_loss += loss.item()
            total_iou += iou
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    return (avg_loss, avg_iou)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', required=True)
    parser.add_argument('--val_json', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output) / f'fold{args.fold}'
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = WS1Dataset(args.train_json, args.data_root)
    val_dataset = WS1Dataset(args.val_json, args.data_root)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = smp.DeepLabV3Plus(encoder_name='resnet101', encoder_weights='imagenet', in_channels=3, classes=1)
    model = model.to(device)
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_iou': best_iou, 'val_loss': val_loss}
            torch.save(checkpoint, output_dir / 'best_model.pth')
if __name__ == '__main__':
    main()
