#!/usr/bin/env python3
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings("ignore")


class InferenceBenchmark:
    def __init__(self, device="cuda", image_size=1024, n_warmup=10, n_runs=100):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        self.device = device
        self.image_size = image_size
        self.n_warmup = n_warmup
        self.n_runs = n_runs

    def benchmark_model(self, model, model_name, prepare_input_fn=None):
        model.eval()
        model.to(self.device)
        x = prepare_input_fn() if prepare_input_fn else torch.randn(1, 3, self.image_size, self.image_size, device=self.device)
        with torch.no_grad():
            for _ in range(self.n_warmup):
                model(x)
                torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(self.n_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        a = np.array(times)
        return {"model_name": model_name, "mean_ms": float(a.mean()), "std_ms": float(a.std()), "min_ms": float(a.min()), "max_ms": float(a.max()), "median_ms": float(np.median(a)), "fps": float(1000 / a.mean()), "n_runs": self.n_runs, "all_timings_ms": a.tolist()}

    def write(self, results, output):
        report = {"benchmark_config": {"device": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda, "pytorch_version": torch.__version__, "image_size": self.image_size, "n_warmup": self.n_warmup, "n_runs": self.n_runs, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "results": results}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_sam_lora(model_ckpt, base_ckpt, device, image_size):
    import torch.nn as nn
    import torch.nn.functional as F
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir.parent / "methods" / "sam_lora" / "scripts"))
    from lora_inject import create_sam_lora_model
    ckpt = torch.load(model_ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    sam_checkpoint = str(base_ckpt) if base_ckpt and base_ckpt.exists() else cfg.get("sam_checkpoint")
    if not sam_checkpoint:
        raise RuntimeError("SAM base checkpoint is required")
    model = create_sam_lora_model(sam_checkpoint=sam_checkpoint, model_type=cfg.get("sam_model", "vit_h"), lora_rank=cfg.get("lora_rank", 8), lora_alpha=cfg.get("lora_alpha", 16), lora_dropout=cfg.get("lora_dropout", 0.05), device=device)
    for key, strict in (("model_state_dict", True), ("model_state_dict", False), ("adapter_state_dict", False)):
        state = ckpt.get(key)
        if state is None:
            continue
        try:
            model.load_state_dict(state, strict=strict)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Could not load {model_ckpt}")

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.sam = model

        def forward(self, x):
            b = x.shape[0]
            emb = self.sam.image_encoder(x)
            coords = torch.full((b, 1, 2), image_size / 2, device=device)
            labels = torch.ones((b, 1), device=device)
            sparse, dense = self.sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
            low_res, _ = self.sam.mask_decoder(image_embeddings=emb, image_pe=self.sam.prompt_encoder.get_dense_pe().expand(b, -1, -1, -1), sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
            return F.interpolate(low_res, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return Wrapper()


def load_deeplab(ckpt_path, device):
    import segmentation_models_pytorch as smp
    model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    return model


def load_classical(model_path, preproc_path, config_path, image_size):
    import torch.nn as nn
    import yaml
    import pickle
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir.parent / "methods" / "classical" / "scripts"))
    from inference_with_crf import XGBoostCRFInference
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    
    # Load preprocessor manually to handle the dict format
    engine = XGBoostCRFInference(str(model_path), cfg, None, cfg.get("features", {}).get("backend", "handcrafted"), cfg.get("features", {}).get("cnn_model", "vgg16"), cfg.get("features", {}).get("patch_size", 64), cfg.get("features", {}).get("stride", 32), cfg.get("features", {}).get("batch_size", 64), 1)
    
    if preproc_path and Path(preproc_path).exists():
        with open(preproc_path, 'rb') as f:
            preproc_data = pickle.load(f)
        # Handle the dict format saved by train_fold_xgb.py
        if isinstance(preproc_data, dict) and 'pipeline' in preproc_data:
            engine.preprocessor = preproc_data['pipeline']
        else:
            # Fallback to original loading logic
            try:
                from preprocessing_pipeline import ClassicalPreprocessor
                engine.preprocessor = ClassicalPreprocessor.load(str(preproc_path))
            except Exception:
                engine.preprocessor = preproc_data

    class Wrapper(nn.Module):
        def forward(self, x):
            img = x[0].detach().cpu().float().numpy()
            img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-6), 0, 1)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1]
            engine.predict_image(img, False)
            return torch.zeros((1,), dtype=torch.float32, device=x.device)
    return Wrapper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="logs/inference_benchmarks.json")
    ap.add_argument("--n-runs", type=int, default=100)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--sam-model-path", type=Path)
    ap.add_argument("--sam-base-path", type=Path)
    ap.add_argument("--deeplab-model-path", type=Path)
    ap.add_argument("--classical-model-path", type=Path)
    ap.add_argument("--classical-preproc-path", type=Path)
    ap.add_argument("--classical-config-path", type=Path)
    args = ap.parse_args()
    bench = InferenceBenchmark("cuda", args.image_size, args.n_warmup, args.n_runs)
    x = lambda: torch.randn(1, 3, args.image_size, args.image_size, device="cuda")
    results = []
    if args.sam_model_path:
        results.append(bench.benchmark_model(load_sam_lora(args.sam_model_path, args.sam_base_path, "cuda", args.image_size), "SAM+LoRA", x))
    if args.deeplab_model_path:
        results.append(bench.benchmark_model(load_deeplab(args.deeplab_model_path, "cuda"), "DeepLabV3+", x))
    if args.classical_model_path and args.classical_config_path:
        results.append(bench.benchmark_model(load_classical(args.classical_model_path, args.classical_preproc_path, args.classical_config_path, args.image_size), "VGG16+XGBoost", x))
    bench.write(results, Path(args.output))


if __name__ == "__main__":
    main()
