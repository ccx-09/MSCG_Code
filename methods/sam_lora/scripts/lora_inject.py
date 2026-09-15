#!/usr/bin/env python3
import sys
import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from segment_anything import sam_model_registry
from segment_anything.modeling.mask_decoder import MaskDecoder

def _patch_mask_decoder_batch_support() -> bool:
    if getattr(MaskDecoder.predict_masks, '_mdp_batch_patch_applied', False):
        return False

    def patched_predict_masks(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        if image_embeddings.shape[0] != dense_prompt_embeddings.shape[0]:
            raise ValueError(f'mask decoder received mismatched batch sizes: image_embeddings={image_embeddings.shape[0]} vs dense_prompt_embeddings={dense_prompt_embeddings.shape[0]}')
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return (masks, iou_pred)
    patched_predict_masks._mdp_batch_patch_applied = True
    MaskDecoder.predict_masks = patched_predict_masks
    logging.getLogger(__name__).info('✅ Patched SAM MaskDecoder for batched prompts')
    return True
_patch_mask_decoder_batch_support()

class LoRALayer(nn.Module):

    def __init__(self, original_layer: nn.Linear, rank: int=8, alpha: float=16.0, dropout: float=0.05, freeze_original: bool=True):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.input_dim = original_layer.in_features
        self.output_dim = original_layer.out_features
        if freeze_original:
            for param in self.original_layer.parameters():
                param.requires_grad = False
        self.lora_A = nn.Linear(self.input_dim, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.output_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        device = next(original_layer.parameters()).device
        self.lora_A = self.lora_A.to(device)
        self.lora_B = self.lora_B.to(device)
        self.dropout = self.dropout.to(device)
        self.lora_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original_layer(x)
        if self.lora_enabled:
            lora_out = self.lora_A(x)
            lora_out = self.dropout(lora_out)
            lora_out = self.lora_B(lora_out)
            lora_out = lora_out * self.scaling
            result = result + lora_out
        return result

    def enable_lora(self):
        self.lora_enabled = True

    def disable_lora(self):
        self.lora_enabled = False

class SAMLoRAInjector:

    def __init__(self, target_layers: List[str]=None, rank: int=8, alpha: float=16.0, dropout: float=0.05, budget_percentage: Optional[float]=None):
        self.target_layers = target_layers or ['attn_q', 'attn_k', 'attn_v', 'attn_o']
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.budget_percentage = budget_percentage
        self.injected_layers = []
        self.original_param_count = 0
        self.trainable_param_count = 0
        self.total_param_count = 0

    def inject_sam_model(self, model) -> Dict[str, int]:
        self.original_param_count = sum((p.numel() for p in model.parameters()))
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'blocks'):
            blocks = model.image_encoder.blocks
            if self.budget_percentage is not None:
                self.rank = self._calculate_rank_for_budget(model, blocks)
            for block_idx, block in enumerate(blocks):
                self._inject_block(block, block_idx)
        self.total_param_count = sum((p.numel() for p in model.parameters()))
        self.trainable_param_count = sum((p.numel() for p in model.parameters() if p.requires_grad))
        trainable_percentage = self.trainable_param_count / self.total_param_count * 100
        budget_met = True
        if self.budget_percentage is not None:
            budget_met = abs(trainable_percentage - self.budget_percentage) < 0.05
        return {'original_params': self.original_param_count, 'total_params': self.total_param_count, 'trainable_params': self.trainable_param_count, 'trainable_percentage': trainable_percentage, 'lora_rank': self.rank, 'target_layers': self.target_layers, 'injected_layers': len(self.injected_layers), 'budget_met': budget_met, 'budget_target': self.budget_percentage}

    def _inject_block(self, block, block_idx: int):
        if hasattr(block, 'attn'):
            attn = block.attn
            if 'attn_q' in self.target_layers and hasattr(attn, 'qkv'):
                self._inject_qkv_layer(attn, block_idx)
            if 'attn_o' in self.target_layers and hasattr(attn, 'proj'):
                original_proj = attn.proj
                lora_proj = LoRALayer(original_proj, self.rank, self.alpha, self.dropout, freeze_original=False)
                attn.proj = lora_proj
                self.injected_layers.append(f'block_{block_idx}_attn_proj')
        if hasattr(block, 'mlp'):
            mlp = block.mlp
            if 'mlp_lin1' in self.target_layers and hasattr(mlp, 'lin1'):
                original_lin1 = mlp.lin1
                lora_lin1 = LoRALayer(original_lin1, self.rank, self.alpha, self.dropout, freeze_original=False)
                mlp.lin1 = lora_lin1
                self.injected_layers.append(f'block_{block_idx}_mlp_lin1')
            if 'mlp_lin2' in self.target_layers and hasattr(mlp, 'lin2'):
                original_lin2 = mlp.lin2
                lora_lin2 = LoRALayer(original_lin2, self.rank, self.alpha, self.dropout, freeze_original=False)
                mlp.lin2 = lora_lin2
                self.injected_layers.append(f'block_{block_idx}_mlp_lin2')

    def _inject_qkv_layer(self, attn, block_idx: int):
        if not hasattr(attn, 'qkv'):
            return
        original_qkv = attn.qkv
        embed_dim = original_qkv.in_features
        if any((layer in self.target_layers for layer in ['attn_q', 'attn_k', 'attn_v'])):
            lora_qkv = LoRALayer(original_qkv, self.rank, self.alpha, self.dropout, freeze_original=False)
            attn.qkv = lora_qkv
            self.injected_layers.append(f'block_{block_idx}_attn_qkv')

    def _calculate_rank_for_budget(self, model, blocks) -> int:
        if self.budget_percentage is None:
            return self.rank
        target_trainable_params = self.original_param_count * (self.budget_percentage / 100)
        num_blocks = len(blocks)
        layers_per_block = len([l for l in self.target_layers if l.startswith('attn') or l.startswith('mlp')])
        embed_dim = 768
        params_per_rank_per_layer = 2 * embed_dim
        total_layers = num_blocks * layers_per_block
        if total_layers > 0:
            estimated_rank = int(target_trainable_params / (total_layers * params_per_rank_per_layer))
            estimated_rank = max(1, min(estimated_rank, 64))
        else:
            estimated_rank = self.rank
        return estimated_rank

def inject_lora_into_sam(model_name: str='vit_b', checkpoint_path: str=None, lora_config: Dict=None, device: str='cpu') -> Tuple[object, Dict]:
    default_config = {'target_layers': ['attn_q', 'attn_k', 'attn_v', 'attn_o'], 'rank': 8, 'alpha': 16.0, 'dropout': 0.05, 'budget_percentage': None}
    if lora_config:
        if 'active_budget' in lora_config:
            active_budget = lora_config['active_budget']
            budget_profiles = lora_config.get('budget_profiles', {'low': 0.1, 'primary': 0.5, 'high': 2.0})
            if active_budget in budget_profiles:
                default_config['budget_percentage'] = budget_profiles[active_budget]
        valid_params = {'target_layers', 'rank', 'alpha', 'dropout', 'budget_percentage'}
        filtered_config = {k: v for k, v in lora_config.items() if k in valid_params}
        default_config.update(filtered_config)
    model = sam_model_registry[model_name](checkpoint=checkpoint_path)
    model = model.to(device)
    injector = SAMLoRAInjector(**default_config)
    stats = injector.inject_sam_model(model)
    return (model, stats)

def create_sam_lora_model(sam_checkpoint: str, model_type: str='vit_h', lora_rank: int=8, lora_alpha: int=16, lora_dropout: float=0.05, device: str='cuda') -> torch.nn.Module:
    lora_config = {'target_layers': ['attn_q', 'attn_k', 'attn_v', 'attn_o'], 'rank': lora_rank, 'alpha': lora_alpha, 'dropout': lora_dropout}
    model, stats = inject_lora_into_sam(model_name=model_type, checkpoint_path=sam_checkpoint, lora_config=lora_config, device=device)
    import logging
    logger = logging.getLogger(__name__)
    print_lora_stats(stats, logger)
    return model

def mark_only_lora_as_trainable(model):
    for param in model.parameters():
        param.requires_grad = False
    lora_params_found = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            module.lora_enabled = True
            for param in module.lora_A.parameters():
                param.requires_grad = True
                lora_params_found += param.numel()
            for param in module.lora_B.parameters():
                param.requires_grad = True
                lora_params_found += param.numel()
    mask_decoder_params = 0
    if hasattr(model, 'mask_decoder'):
        for param in model.mask_decoder.parameters():
            param.requires_grad = True
            mask_decoder_params += param.numel()
    trainable_count = sum((p.numel() for p in model.parameters() if p.requires_grad))

def print_lora_stats(stats: Dict, logger=None):
    if logger is None:
        print_fn = print
    else:
        print_fn = lambda *args, **kwargs: None
    print_fn('=== LoRA Injection Statistics ===')
    print_fn(f"Original parameters: {stats['original_params']:,}")
    print_fn(f"Total parameters: {stats['total_params']:,}")
    print_fn(f"Trainable parameters: {stats['trainable_params']:,}")
    print_fn(f"Trainable percentage: {stats['trainable_percentage']:.3f}%")
    print_fn(f"LoRA rank: {stats['lora_rank']}")
    print_fn(f"Target layers: {stats['target_layers']}")
    print_fn(f"Injected layers: {stats['injected_layers']}")
    print_fn(f"Budget constraint met: {stats['budget_met']}")
