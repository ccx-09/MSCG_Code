#!/usr/bin/env python3
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
sys.path.insert(0, str(Path(__file__).parent))
from modern_dataset import ModernSAMDataset
from lora_inject import create_sam_lora_model
logger = logging.getLogger(__name__)

class SAMFoldInference:

    def __init__(self, model_path: str, ws1_root: str, fold: int, policy: str='P1', device: str='cuda'):
        self.model_path = Path(model_path)
        self.ws1_root = Path(ws1_root)
        self.fold = fold
        self.policy = policy
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f'Model not found: {self.model_path}')
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        sam_checkpoint = self.config.get('sam_checkpoint')
        if not sam_checkpoint or not Path(sam_checkpoint).exists():
            sam_checkpoint = 'sam_vit_h_4b8939.pth'
        self.model = create_sam_lora_model(sam_checkpoint=sam_checkpoint, model_type=self.config.get('sam_model', 'vit_h'), lora_rank=self.config.get('lora_rank', 8), lora_alpha=self.config.get('lora_alpha', 16), lora_dropout=self.config.get('lora_dropout', 0.05), device=self.device)
        loaded = False
        err: Optional[Exception] = None
        for key, strict_flag in [('model_state_dict', True), ('model_state_dict', False), ('adapter_state_dict', False)]:
            state = checkpoint.get(key)
            if state is None:
                continue
            try:
                missing, unexpected = self.model.load_state_dict(state, strict=strict_flag)
                if strict_flag is False:
                    pass
                loaded = True
                break
            except Exception as e:
                err = e
                continue
        if not loaded:
            raise RuntimeError(f'Failed to load model weights from checkpoint: {err}')
        self.model.eval()

    def setup_dataset(self):
        manifest_dir = self.ws1_root / 'folds'
        test_json = manifest_dir / f'fold{self.fold}_test.json'
        if not test_json.exists():
            raise FileNotFoundError(f'Test manifest not found: {test_json}')
        self.test_dataset = ModernSAMDataset(coco_json=str(test_json), data_root=str(self.ws1_root), prompt_policy=self.policy, img_size=1024, augment=False, split='test')
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    def predict_batch(self, batch: Dict) -> torch.Tensor:
        images = batch['image'].to(self.device)
        prompts = {k: v.to(self.device) for k, v in batch['prompts'].items()}
        with torch.no_grad():
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            sam_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).div(255.0).view(1, 3, 1, 1)
            sam_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).div(255.0).view(1, 3, 1, 1)
            img01 = images * imagenet_std + imagenet_mean
            images_sam = (img01 - sam_mean) / sam_std
            image_embeddings = self.model.image_encoder(images_sam)
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=(prompts['points'], prompts['labels']), boxes=None, masks=None)
            low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=image_embeddings, image_pe=self.model.prompt_encoder.get_dense_pe().expand(images.size(0), -1, -1, -1), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=True)
            best_idx = torch.argmax(iou_predictions, dim=1)
            low_res_masks = torch.stack([low_res_masks[i, best_idx[i]:best_idx[i] + 1] for i in range(len(best_idx))], dim=0)
            pred_masks = F.interpolate(low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False)
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
        return pred_binary

    def run_inference(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.ws1_root / 'folds' / f'fold{self.fold}_test.json'
        with open(manifest_path, 'r') as f:
            test_manifest = json.load(f)
        image_to_sample: Dict[Any, Dict[str, Any]] = {}
        for img_info in test_manifest['images']:
            img_id = img_info.get('id')
            if img_id is not None:
                image_to_sample[img_id] = img_info
                image_to_sample[str(img_id)] = img_info
            file_name = img_info.get('file_name', '')
            if file_name:
                try:
                    image_to_sample[Path(file_name).stem] = img_info
                except Exception:
                    pass
        results = []

        def _as_scalar(x: Any) -> Any:
            try:
                import torch
                import numpy as np
            except Exception:
                torch = None
                np = None
            if isinstance(x, (list, tuple)) and len(x) == 1:
                x = x[0]
            if torch is not None and isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return x.item()
                x = x.detach().cpu().tolist()
                if isinstance(x, (list, tuple)) and len(x) == 1:
                    return x[0]
                return x
            if np is not None and isinstance(x, np.ndarray):
                if x.size == 1:
                    return x.item()
                return x.tolist()
            return x
        for idx, batch in enumerate(tqdm(self.test_loader, desc='Predicting')):
            pred_mask = self.predict_batch(batch)
            pred_np = pred_mask[0, 0].cpu().numpy()
            pred_uint8 = (pred_np * 255).astype(np.uint8)
            meta = batch['meta']
            sample_id = _as_scalar(meta.get('sample_id'))
            img_info = image_to_sample.get(sample_id)
            if img_info is None:
                img_info = image_to_sample.get(str(sample_id))
            if img_info is None and sample_id is not None:
                img_info = image_to_sample.get(Path(str(sample_id)).stem)
            if img_info is None:
                img_info = {'file_name': f'unknown_{idx}.jpg'}
            uuid = Path(img_info['file_name']).stem
            corruption = _as_scalar(meta.get('corruption'))
            scale_bin = _as_scalar(meta.get('scale_bin'))
            severity = _as_scalar(meta.get('severity'))
            try:
                severity_int = int(severity)
            except Exception:
                s = str(severity)
                digits = ''.join((ch for ch in s if ch.isdigit()))
                severity_int = int(digits) if digits else 0
            flat_path = output_dir / f'{uuid}.png'
            Image.fromarray(pred_uint8).save(flat_path)
            output_subdir = output_dir / str(corruption) / f'{scale_bin}_k{severity_int}'
            output_subdir.mkdir(parents=True, exist_ok=True)
            nested_path = output_subdir / f'{uuid}.png'
            try:
                Image.fromarray(pred_uint8).save(nested_path)
            except Exception:
                pass
            results.append({'sample_id': str(sample_id), 'uuid': uuid, 'corruption': corruption, 'scale_bin': scale_bin, 'severity': int(severity), 'output_path_flat': str(flat_path.name), 'output_path_nested': str(nested_path.relative_to(output_dir)) if 'nested_path' in locals() else None})
        manifest_out = output_dir / 'inference_manifest.json'
        with open(manifest_out, 'w') as f:
            json.dump({'fold': self.fold, 'policy': self.policy, 'num_predictions': len(results), 'predictions': results}, f, indent=2)
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--ws1_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--policy', choices=['P1', 'P2'], default='P1')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    model_path = model_dir / 'best_model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f'Best model not found: {model_path}')
    inferencer = SAMFoldInference(model_path=str(model_path), ws1_root=args.ws1_root, fold=args.fold, policy=args.policy, device=args.device)
    inferencer.setup_dataset()
    results = inferencer.run_inference(Path(args.output_dir))
if __name__ == '__main__':
    main()
