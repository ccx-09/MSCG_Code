#!/usr/bin/env python3
import torch
import time
import numpy as np
import json
from pathlib import Path
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent / 'methods' / 'sam_lora' / 'scripts'))

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0

def benchmark_lora_rank(rank: int, checkpoint_path: Path, sam_base_path: Path, device: str='cuda', image_size: int=1024, n_warmup: int=10, n_runs: int=50) -> dict:
    from lora_inject import create_sam_lora_model
    import torch.nn.functional as F
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config = ckpt.get('config', {})
    model = create_sam_lora_model(sam_checkpoint=str(sam_base_path), model_type=config.get('sam_model', 'vit_h'), lora_rank=rank, lora_alpha=config.get('lora_alpha', rank), lora_dropout=config.get('lora_dropout', 0.05), device='cpu')
    loaded = False
    for key in ['model_state_dict', 'adapter_state_dict']:
        if key in ckpt:
            try:
                model.load_state_dict(ckpt[key], strict=False)
                loaded = True
                break
            except Exception as e:
                continue
    if not loaded:
        raise RuntimeError(f'Failed to load weights from {checkpoint_path}')
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    total_params = sum((p.numel() for p in model.parameters()))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.to(device)
    model.eval()
    sample_input = torch.randn(1, 3, image_size, image_size, device=device)
    baseline_vram_mb = get_gpu_memory_mb()
    with torch.no_grad():
        for i in range(n_warmup):
            image_embeddings = model.image_encoder(sample_input)
            coords = torch.tensor([[image_size / 2.0, image_size / 2.0]], device=device).unsqueeze(0)
            labels = torch.ones((1, 1), device=device)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
            low_res_masks, _ = model.mask_decoder(image_embeddings=image_embeddings, image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False)
            masks = F.interpolate(low_res_masks, size=(image_size, image_size), mode='bilinear', align_corners=False)
            torch.cuda.synchronize()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    vram_gb = peak_vram_mb / 1024.0
    timings = []
    with torch.no_grad():
        for i in range(n_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            image_embeddings = model.image_encoder(sample_input)
            coords = torch.tensor([[image_size / 2.0, image_size / 2.0]], device=device).unsqueeze(0)
            labels = torch.ones((1, 1), device=device)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
            low_res_masks, _ = model.mask_decoder(image_embeddings=image_embeddings, image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False)
            masks = F.interpolate(low_res_masks, size=(image_size, image_size), mode='bilinear', align_corners=False)
            torch.cuda.synchronize()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            timings.append(elapsed_ms)
            if (i + 1) % 10 == 0:
                pass
    timings_array = np.array(timings)
    mean_ms = float(np.mean(timings_array))
    std_ms = float(np.std(timings_array))
    throughput = float(1000.0 / mean_ms)
    del model, sample_input
    torch.cuda.empty_cache()
    return {'rank': rank, 'vram_gb': round(vram_gb, 2), 'throughput_imgs_per_sec': round(throughput, 2), 'mean_ms_per_image': round(mean_ms, 1), 'std_ms_per_image': round(std_ms, 1), 'trainable_params_m': round(trainable_params / 1000000.0, 2), 'trainable_params_pct': round(trainable_params / total_params * 100, 2), 'total_params_m': round(total_params / 1000000.0, 2), 'n_runs': n_runs}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-root', type=str, required=True, help='Path to lora_ablations directory containing lora_r{4,8,16}/fold{0}/best_model.pt')
    parser.add_argument('--sam-base', type=str, required=True, help='Path to SAM base checkpoint sam_vit_h_4b8939.pth')
    parser.add_argument('--ranks', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output', type=str, default='analysis/lora_efficiency_benchmarks.json')
    parser.add_argument('--n-runs', type=int, default=50)
    parser.add_argument('--n-warmup', type=int, default=10)
    parser.add_argument('--image-size', type=int, default=1024)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available - benchmarking requires GPU')
    results = []
    models_root = Path(args.models_root)
    sam_base = Path(args.sam_base)
    for rank in args.ranks:
        checkpoint_dir = models_root / f'lora_r{rank}' / f'fold{args.fold}'
        checkpoint_path = checkpoint_dir / 'best_model.pt'
        if not checkpoint_path.exists():
            continue
        try:
            result = benchmark_lora_rank(rank=rank, checkpoint_path=checkpoint_path, sam_base_path=sam_base, device='cuda', image_size=args.image_size, n_warmup=args.n_warmup, n_runs=args.n_runs)
            results.append(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'benchmark_config': {'device': torch.cuda.get_device_name(0), 'cuda_version': torch.version.cuda, 'pytorch_version': torch.__version__, 'image_size': args.image_size, 'n_warmup': args.n_warmup, 'n_runs': args.n_runs, 'fold': args.fold, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, 'results': results}
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    for r in results:
        pass
if __name__ == '__main__':
    main()
