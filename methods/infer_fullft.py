#!/usr/bin/env python3
import os, sys, json, logging, argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from modern_dataset import ModernSAMDataset
from segment_anything import sam_model_registry

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--ws1_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--policy', choices=['P1', 'P2'], default='P1')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model_dir = Path(args.model_dir)
    model_path = model_dir / 'best_model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f'Full FT model not found: {model_path}')

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})

    sam_checkpoint = config.get('sam_checkpoint', 'sam_vit_h_4b8939.pth')
    model = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    ws1_root = Path(args.ws1_root)
    test_json = ws1_root / 'folds' / f'fold{args.fold}_test.json'
    dataset = ModernSAMDataset(
        coco_json=str(test_json), data_root=str(ws1_root),
        prompt_policy=args.policy, img_size=1024, augment=False, split='test'
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(test_json, 'r') as f:
        test_manifest = json.load(f)

    image_to_sample = {}
    for img_info in test_manifest['images']:
        img_id = img_info.get('id')
        if img_id is not None:
            image_to_sample[img_id] = img_info
            image_to_sample[str(img_id)] = img_info
        file_name = img_info.get('file_name', '')
        if file_name:
            image_to_sample.get(Path(file_name).stem, img_info)

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    sam_mean = torch.tensor([123.675, 116.28, 103.53], device=device).div(255.0).view(1, 3, 1, 1)
    sam_std = torch.tensor([58.395, 57.12, 57.375], device=device).div(255.0).view(1, 3, 1, 1)

    def _to_scalar(x):
        if isinstance(x, torch.Tensor):
            return x.item() if x.numel() == 1 else x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.item() if x.size == 1 else x.tolist()
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return x[0]
        return x

    results = []
    for batch in tqdm(loader, desc=f'Full FT infer fold {args.fold}'):
        images = batch['image'].to(device)
        prompts = {k: v.to(device) for k, v in batch['prompts'].items()}

        with torch.no_grad():
            img01 = images * imagenet_std + imagenet_mean
            images_sam = (img01 - sam_mean) / sam_std

            image_embeddings = model.image_encoder(images_sam)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(prompts['points'], prompts['labels']), boxes=None, masks=None
            )
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe().expand(images.size(0), -1, -1, -1),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            best_idx = torch.argmax(iou_predictions, dim=1)
            low_res_masks = torch.stack([
                low_res_masks[i, best_idx[i]:best_idx[i] + 1] for i in range(len(best_idx))
            ], dim=0)
            pred_masks = F.interpolate(low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False)
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()

        pred_np = pred_binary[0, 0].cpu().numpy()
        pred_uint8 = (pred_np * 255).astype(np.uint8)

        meta = batch['meta']
        sample_id = _to_scalar(meta.get('sample_id'))
        img_info = image_to_sample.get(sample_id) or image_to_sample.get(str(sample_id)) or {}
        uuid = Path(str(img_info.get('file_name', f'unknown_{len(results)}'))).stem

        corruption = _to_scalar(meta.get('corruption', 'unknown'))
        scale_bin = _to_scalar(meta.get('scale_bin', 's3'))
        severity = int(_to_scalar(meta.get('severity', 0)))

        Image.fromarray(pred_uint8).save(output_dir / f'{uuid}.png')

        nested_dir = output_dir / str(corruption) / f'{scale_bin}_k{severity}'
        nested_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred_uint8).save(nested_dir / f'{uuid}.png')

        results.append({
            'sample_id': str(sample_id),
            'uuid': uuid,
            'corruption': str(corruption),
            'scale_bin': str(scale_bin),
            'severity': severity,
            'output_path_flat': f'{uuid}.png',
            'output_path_nested': f'{str(corruption)}/{scale_bin}_k{severity}/{uuid}.png'
        })

    manifest = {
        'fold': args.fold, 'policy': args.policy,
        'num_predictions': len(results), 'predictions': results
    }
    with open(output_dir / 'inference_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f'Full FT inference fold {args.fold} done: {len(results)} predictions')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
