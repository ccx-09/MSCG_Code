#!/usr/bin/env python3
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

class WS1InferenceDataset(Dataset):

    def __init__(self, json_path, data_root):
        with open(json_path) as f:
            data = json.load(f)
        self.images = data['images']
        self.data_root = Path(data_root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_uuid = Path(img_info['file_name']).stem
        img_path = self.data_root / 'composites' / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size
        image = image.resize((512, 512), Image.BILINEAR)
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return (image, img_id, img_uuid, orig_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_json', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = smp.DeepLabV3Plus(encoder_name='resnet101', encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    dataset = WS1InferenceDataset(args.test_json, args.data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for images, img_ids, img_uuids, orig_sizes in tqdm(dataloader, desc='Inference'):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            for i, (pred, img_uuid) in enumerate(zip(preds, img_uuids)):
                pred_np = pred.squeeze().cpu().numpy()
                orig_w, orig_h = (orig_sizes[0][i].item(), orig_sizes[1][i].item())
                pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
                pred_img = pred_img.resize((orig_w, orig_h), Image.NEAREST)
                output_path = output_dir / f'{img_uuid}.png'
                pred_img.save(output_path)
if __name__ == '__main__':
    main()
