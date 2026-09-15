#!/bin/bash
set -eo pipefail

# ============================================================
# Full-parameter Fine-tuning vs LoRA — Section 6.3 Experiment
#
# Compares: Full FT and LoRA r=4, r=8, r=16 using the five-fold protocol.
# Existing r=8 results can be reused from the three-way experiment cache.
#
# Usage:
#   bash run_fullft_lora.sh                            # Run all
#   bash run_fullft_lora.sh --phase fullft_train       # Single phase
#   bash run_fullft_lora.sh --phase lora_train --fold 0
#
# Optional path variables (defaults are relative to this script):
#   PROJECT_ROOT, DATA_ROOT, SAM_CKPT, CACHE_DIR, OUTPUT_DIR
# Example:
#   DATA_ROOT=/path/to/MSCG_dataset \
#   SAM_CKPT=/path/to/sam_vit_h_4b8939.pth \
#   bash run_fullft_lora.sh --phase eval
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
METHODS_DIR="${METHODS_DIR:-$PROJECT_ROOT/methods}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$PROJECT_ROOT/analysis}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$PROJECT_ROOT/artifacts}"
DATA_ROOT="${DATA_ROOT:-${MSCG_DATA_ROOT:-}}"
SAM_CKPT="${SAM_CKPT:-${SAM_CHECKPOINT:-}}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/cache/mscg_fullft}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACTS_DIR/newdata/fullft_lora}"
LOG_DIR="$CACHE_DIR/logs"
EXISTING_5FOLD="${EXISTING_5FOLD:-$ARTIFACTS_DIR/newdata/ws1_kfold}"
EXISTING_CACHE="${EXISTING_CACHE:-$PROJECT_ROOT/cache/mscg}"

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_DIR"

log()   { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/experiment.log"; }
section() { echo "" | tee -a "$LOG_DIR/experiment.log"; log "========== $1 =========="; }

# --- Arg parsing ---
RUN_ALL=true
SPECIFIC_PHASE=""
SPECIFIC_FOLD=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)  RUN_ALL=false; SPECIFIC_PHASE="$2"; shift 2 ;;
        --fold)   SPECIFIC_FOLD="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -n "$SPECIFIC_FOLD" ]; then
    FOLD_RANGE="$SPECIFIC_FOLD"
else
    FOLD_RANGE=$(seq 0 4)
fi

run_phase() {
    if $RUN_ALL || [ "$SPECIFIC_PHASE" == "$1" ]; then return 0; else return 1; fi
}

save_training_meta() {
    local config=$1 fold=$2 out_dir=$3 start_ts=$4 end_ts=$5
    local ckpt_file="$out_dir/best_model.pt"
    local ckpt_size=0
    [ -f "$ckpt_file" ] && ckpt_size=$(stat --format=%s "$ckpt_file")
    local elapsed=$((end_ts - start_ts))

    local trainable_params trainable_pct total_params
    if [ "$config" = "full_ft" ]; then
        trainable_params=641 total_params=641 trainable_pct=100.0
    else
        trainable_params=$(python3 -c "
import json, re, sys
try:
    with open('$out_dir/config.json') as f:
        c = json.load(f)
    r = c.get('lora_rank', 8)
    # ViT-H: ~641M total; LoRA adds rank*2*dim per target module
    # Approx: ~0.79% for r=4, ~0.94% for r=8, ~1.25% for r=16
    pcts = {4: 0.79, 8: 0.94, 16: 1.25}
    pct = pcts.get(r, 0.8)
    tp = round(641 * pct / 100, 1)
    print(f'{tp}|{pct}')
except: print('6|0.94')
" 2>/dev/null)
        trainable_params=$(echo "$trainable_params" | cut -d'|' -f1)
        trainable_pct=$(echo "$trainable_params" | cut -d'|' -f2)
        total_params=641
    fi

    cat > "$out_dir/training_meta.json" <<EOF
{
  "config": "$config",
  "fold": $fold,
  "training_time_seconds": $elapsed,
  "checkpoint_size_bytes": $ckpt_size,
  "best_val_iou": $(python3 -c "
import json
h = json.load(open('$out_dir/training_history.json'))
best = max(h, key=lambda x: x['val']['iou']) if h else {}
print(best.get('val', {}).get('iou', 0))
" 2>/dev/null || echo 0),
  "trainable_params_m": $trainable_params,
  "trainable_param_pct": $trainable_pct,
  "total_params_m": $total_params
}
EOF
}

# ============================================================
# Phase 0: Prerequisites
# ============================================================
phase_prereqs() {
    section "Phase 0: Prerequisites"

    [ ! -d "$DATA_ROOT/composites" ] && { echo "  Missing data"; exit 1; } || echo "  OK: data"
    [ ! -f "$SAM_CKPT" ] && { echo "  Missing SAM"; exit 1; } || echo "  OK: SAM"
    python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null || { echo "  No CUDA"; exit 1; }
    echo "  OK: CUDA"

    for fold in 0 1 2 3 4; do
        for split in train val test; do
            [ ! -f "$DATA_ROOT/folds/fold${fold}_${split}.json" ] && { echo "  Missing fold${fold}_${split}.json"; exit 1; }
        done
    done
    echo "  OK: fold JSONs"
    echo "  Cache: $CACHE_DIR"
    echo "  Output: $OUTPUT_DIR"
}

# ============================================================
# Phase 1: Full FT Training (folds 0-2)
# ============================================================
phase_fullft_train() {
    section "Phase 1: Full FT Training (3 folds)"

    local folds_to_run
    [ -n "$SPECIFIC_FOLD" ] && folds_to_run="$SPECIFIC_FOLD" || folds_to_run="0 1 2"

    for fold in $folds_to_run; do
        local out_dir="$CACHE_DIR/full_ft/fold$fold"
        if [ -f "$out_dir/training_complete.json" ]; then
            log "  skip full_ft fold $fold (completed)"
            continue
        fi
        log "  >> Full FT fold $fold"
        mkdir -p "$out_dir"
        local start_ts
        start_ts=$(date +%s)
        python "$METHODS_DIR/train_full_ft_sam.py" \
            --fold $fold \
            --ws1_root "$DATA_ROOT" \
            --sam_checkpoint "$SAM_CKPT" \
            --output_dir "$out_dir" \
            2>&1 | tee -a "$LOG_DIR/fullft_fold${fold}.log"
        local end_ts
        end_ts=$(date +%s)
        save_training_meta "full_ft" "$fold" "$out_dir" "$start_ts" "$end_ts"
        log "  done full_ft fold $fold"
    done
}

# ============================================================
# Phase 2: LoRA rank ablation training
# ============================================================
phase_lora_train() {
    local rank=$1 label=$2
    section "Phase 2${label}: LoRA r=${rank} Training (5 folds)"

    for fold in $FOLD_RANGE; do
        local out_dir="$CACHE_DIR/${label}/fold$fold"
        if [ -f "$out_dir/best_model.pt" ]; then
            log "  skip ${label} fold $fold (exists)"
            continue
        fi
        log "  >> ${label} fold $fold"
        mkdir -p "$out_dir"
        local start_ts
        start_ts=$(date +%s)
        python "$METHODS_DIR/train_fold_sam.py" \
            --fold $fold \
            --ws1_root "$DATA_ROOT" \
            --sam_checkpoint "$SAM_CKPT" \
            --policy P1 --epochs 50 \
            --batch_size 4 --gradient_accumulation_steps 4 \
            --lr 5e-5 --early_stopping_patience 3 \
            --lora_rank $rank --lora_alpha $((rank * 2)) \
            --output_dir "$out_dir" \
            2>&1 | tee -a "$LOG_DIR/${label}_fold${fold}.log"
        local end_ts
        end_ts=$(date +%s)
        save_training_meta "${label}" "$fold" "$out_dir" "$start_ts" "$end_ts"
        log "  done ${label} fold $fold"
    done
}

# ============================================================
# Phase 3: Inference
# ============================================================
phase_infer() {
    section "Phase 3: Inference"

    # --- Full FT inference (folds 0-2) ---
    local ft_folds
    [ -n "$SPECIFIC_FOLD" ] && ft_folds="$SPECIFIC_FOLD" || ft_folds="0 1 2"
    for fold in $ft_folds; do
        local pred_dir="$CACHE_DIR/pred_full_ft/fold$fold"
        if [ -f "$pred_dir/inference_manifest.json" ]; then
            log "  skip infer full_ft fold $fold (exists)"
            continue
        fi
        local model_dir="$CACHE_DIR/full_ft/fold$fold"
        [ ! -f "$model_dir/best_model.pt" ] && { log "  no model full_ft fold $fold, skip"; continue; }
        log "  >> infer full_ft fold $fold"
        mkdir -p "$pred_dir"
        python "$METHODS_DIR/infer_fullft.py" \
            --fold $fold \
            --model_dir "$model_dir" \
            --ws1_root "$DATA_ROOT" \
            --output_dir "$pred_dir" \
            --policy P1 \
            2>&1 | tee -a "$LOG_DIR/infer_fullft_fold${fold}.log"
        log "  done infer full_ft fold $fold"
    done

    # --- LoRA inference (r=4, r=16, 5 folds) ---
    for combo in "lora_r4:4" "lora_r16:16"; do
        local label="${combo%%:*}"
        for fold in $FOLD_RANGE; do
            local pred_dir="$CACHE_DIR/pred_${label}/fold$fold"
            if [ -f "$pred_dir/inference_manifest.json" ]; then
                log "  skip infer ${label} fold $fold (exists)"
                continue
            fi
            local model_dir="$CACHE_DIR/${label}/fold$fold"
            [ ! -f "$model_dir/best_model.pt" ] && { log "  no model ${label} fold $fold, skip"; continue; }
            log "  >> infer ${label} fold $fold"
            mkdir -p "$pred_dir"
            python "$METHODS_DIR/infer_fold_sam.py" \
                --fold $fold \
                --model_dir "$model_dir" \
                --ws1_root "$DATA_ROOT" \
                --output_dir "$pred_dir" \
                --policy P1 \
                2>&1 | tee -a "$LOG_DIR/infer_${label}_fold${fold}.log"
            log "  done infer ${label} fold $fold"
        done
    done
}

# ============================================================
# Phase 4: Evaluation
# ============================================================
phase_eval() {
    section "Phase 4: Granular Evaluation"

    local configs=("full_ft:0 1 2" "lora_r4:0 1 2 3 4" "lora_r16:0 1 2 3 4")
    for entry in "${configs[@]}"; do
        local label="${entry%%:*}"
        local folds_to_run="${entry#*:}"
        [ -n "$SPECIFIC_FOLD" ] && folds_to_run="$SPECIFIC_FOLD"

        for fold in $folds_to_run; do
            local pred_dir="$CACHE_DIR/pred_${label}/fold$fold"
            local out_dir="$OUTPUT_DIR/${label}/fold$fold"
            if [ -f "$out_dir/summary.json" ]; then
                log "  skip eval ${label} fold $fold (exists)"
                continue
            fi
            if [ ! -d "$pred_dir" ] || [ -z "$(ls -A "$pred_dir" 2>/dev/null)" ]; then
                log "  no predictions ${label} fold $fold, skip"
                continue
            fi
            local test_json="$DATA_ROOT/folds/fold${fold}_test.json"
            log "  >> eval ${label} fold $fold"
            mkdir -p "$out_dir"
            python "$ANALYSIS_DIR/ws1_granular_eval.py" \
                --ws1-json "$test_json" \
                --pred-dir "$pred_dir" \
                --gt-dir "$DATA_ROOT/masks" \
                --out-dir "$out_dir" \
                2>&1 | tee -a "$LOG_DIR/eval_${label}_fold${fold}.log"
            log "  done eval ${label} fold $fold"
        done
    done
}

# ============================================================
# Phase 5: Aggregate
# ============================================================
phase_aggregate() {
    section "Phase 5: Aggregating Results"

    # Copy existing r=8 eval results from 5-fold experiment
    if [ -d "$EXISTING_5FOLD" ]; then
        for fold in 0 1 2 3 4; do
            local src="$EXISTING_5FOLD/fold$fold/sam/summary.json"
            local dst="$OUTPUT_DIR/lora_r8/fold$fold/summary.json"
            if [ -f "$src" ] && [ ! -f "$dst" ]; then
                mkdir -p "$(dirname "$dst")"
                cp "$src" "$dst"
                log "  copied r8 fold $fold from 5-fold eval"
            fi
        done
        # Also copy training meta from existing cache
        for fold in 0 1 2 3 4; do
            local src_meta="$EXISTING_CACHE/sam/fold$fold/config.json"
            local dst_meta="$OUTPUT_DIR/lora_r8/fold$fold/"
            if [ -f "$src_meta" ] && [ ! -f "$dst_meta/training_meta.json" ]; then
                mkdir -p "$dst_meta"
                python3 -c "
import json, os
c = json.load(open('$src_meta'))
r = c.get('lora_rank', 8)
ckpt = '$EXISTING_CACHE/sam/fold$fold/best_model.pt'
sz = os.path.getsize(ckpt) if os.path.exists(ckpt) else 0
pcts = {4: 0.79, 8: 0.94, 16: 1.25}
pct = pcts.get(r, 0.94)
tp = round(641 * pct / 100, 1)
meta = {
    'config': 'lora_r8', 'fold': $fold, 'training_time_seconds': 0,
    'checkpoint_size_bytes': sz, 'best_val_iou': 0,
    'trainable_params_m': tp, 'trainable_param_pct': pct, 'total_params_m': 641
}
with open('$dst_meta/training_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
                log "  copied r8 fold $fold training meta"
            fi
        done
    fi

    # Generate aggregated comparison JSON
    log "  >> generating fullft_lora_comparison.json"
    python3 -c "
import json, numpy as np
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
configs = ['full_ft', 'lora_r4', 'lora_r8', 'lora_r16']
comparison = {}

for cfg in configs:
    cfg_dir = output_dir / cfg
    if not cfg_dir.exists():
        continue
    fold_dirs = sorted(cfg_dir.glob('fold*'))
    test_mious = []
    test_dices = []
    val_mious = []
    train_times = []
    ckpt_sizes = []
    tp_m = None
    tp_pct = None
    n_folds = 0

    for fd in fold_dirs:
        # test metrics from eval
        summary_file = fd / 'summary.json'
        if summary_file.exists():
            s = json.load(open(summary_file))
            test_mious.append(s['overall']['miou'])
            test_dices.append(s['overall']['dice'])
            n_folds += 1

        # training meta
        meta_file = fd / 'training_meta.json'
        if meta_file.exists():
            m = json.load(open(meta_file))
            if m.get('best_val_iou', 0) > 0:
                val_mious.append(m['best_val_iou'])
            train_times.append(m.get('training_time_seconds', 0))
            ckpt_sizes.append(m.get('checkpoint_size_bytes', 0))
            if tp_m is None:
                tp_m = m.get('trainable_params_m', 0)
                tp_pct = m.get('trainable_param_pct', 0)

    if not test_mious:
        continue

    mean_time = np.mean(train_times) if train_times else 0
    mean_ckpt = np.mean(ckpt_sizes) if ckpt_sizes else 0

    comparison[cfg] = {
        'num_folds': n_folds,
        'test_miou': {
            'mean': float(np.mean(test_mious)),
            'std': float(np.std(test_mious, ddof=1)) if len(test_mious) > 1 else 0,
            'values': [float(v) for v in test_mious]
        },
        'test_dice': {
            'mean': float(np.mean(test_dices)),
            'std': float(np.std(test_dices, ddof=1)) if len(test_dices) > 1 else 0,
            'values': [float(v) for v in test_dices]
        },
        'val_miou': {
            'mean': float(np.mean(val_mious)) if val_mious else 0,
            'std': float(np.std(val_mious, ddof=1)) if len(val_mious) > 1 else 0
        },
        'trainable_params_m': tp_m or 0,
        'trainable_param_pct': tp_pct or 0,
        'total_params_m': 641,
        'training_time_hours': round(mean_time / 3600, 2),
        'checkpoint_size_mb': round(mean_ckpt / (1024 * 1024), 1)
    }

    # Add per-fold summary
    comparison[cfg]['per_fold'] = []
    for fd in sorted(cfg_dir.glob('fold*')):
        summary_file = fd / 'summary.json'
        meta_file = fd / 'training_meta.json'
        entry = {'fold': int(fd.name.replace('fold', ''))}
        if summary_file.exists():
            s = json.load(open(summary_file))
            entry['test_miou'] = s['overall']['miou']
            entry['test_dice'] = s['overall']['dice']
            entry['per_scale_miou'] = s.get('per_scale_miou', {})
        if meta_file.exists():
            m = json.load(open(meta_file))
            entry['training_time_s'] = m.get('training_time_seconds', 0)
            entry['checkpoint_size_bytes'] = m.get('checkpoint_size_bytes', 0)
            entry['best_val_iou'] = m.get('best_val_iou', 0)
        comparison[cfg]['per_fold'].append(entry)

# Rankings
ranked = sorted(comparison.keys(), key=lambda c: comparison[c]['test_miou']['mean'], reverse=True)
rankings = []
for r, cfg in enumerate(ranked, 1):
    rankings.append({'config': cfg, 'rank': r, 'miou': comparison[cfg]['test_miou']['mean']})

output = {
    'configs': list(comparison.keys()),
    'method_order': ['full_ft', 'lora_r16', 'lora_r8', 'lora_r4'],
    'comparison': comparison,
    'rankings': rankings,
    'summary': {
        'best_config': ranked[0] if ranked else None,
        'best_miou': comparison[ranked[0]]['test_miou']['mean'] if ranked else 0,
        'full_ft_folds': 3,
        'lora_folds': 5
    }
}

with open(output_dir / 'fullft_lora_comparison.json', 'w') as f:
    json.dump(output, f, indent=2)

print('  Configs:', list(comparison.keys()))
for cfg in comparison:
    m = comparison[cfg]
    print(f'  {cfg}: mIoU={m[\"test_miou\"][\"mean\"]*100:.2f}% ± {m[\"test_miou\"][\"std\"]*100:.2f}% ({m[\"num_folds\"]} folds)')
"
}

# ============================================================
# Phase Summary
# ============================================================
phase_summary() {
    section "Experiment Summary"
    echo "  Cache:     $CACHE_DIR"
    echo "  Output:    $OUTPUT_DIR"
    echo "  Logs:      $LOG_DIR"
    echo ""

    echo "  === Training Status ==="
    for cfg in full_ft lora_r4 lora_r16; do
        echo ""
        echo "  $cfg:"
        local folds
        [ "$cfg" = "full_ft" ] && folds="0 1 2" || folds="0 1 2 3 4"
        for fold in $folds; do
            local ckpt="$CACHE_DIR/$cfg/fold$fold/best_model.pt"
            local meta="$CACHE_DIR/$cfg/fold$fold/training_meta.json"
            local status=" "
            [ -f "$ckpt" ] && status="T" || status=" "
            local time_str=""
            [ -f "$meta" ] && time_str=$(python3 -c "import json; m=json.load(open('$meta')); print(f'{m[\"training_time_seconds\"]//60}m')" 2>/dev/null || echo "")
            echo "    Fold $fold: [$status] $time_str"
        done
    done

    echo ""
    echo "  === Eval Status ==="
    for cfg in full_ft lora_r4 lora_r8 lora_r16; do
        echo "  $cfg:"
        local folds
        [ "$cfg" = "full_ft" ] && folds="0 1 2" || folds="0 1 2 3 4"
        for fold in $folds; do
            local sf="$OUTPUT_DIR/$cfg/fold$fold/summary.json"
            local status=" "
            [ -f "$sf" ] && status="E" || status=" "
            echo "    Fold $fold: [$status]"
        done
    done

    if [ -f "$OUTPUT_DIR/fullft_lora_comparison.json" ]; then
        echo ""
        python3 -c "
import json
d = json.load(open('$OUTPUT_DIR/fullft_lora_comparison.json'))
for cfg in d.get('method_order', []):
    if cfg not in d.get('comparison', {}): continue
    m = d['comparison'][cfg]
    print(f'  {cfg}: mIoU = {m[\"test_miou\"][\"mean\"]*100:.2f}% ± {m[\"test_miou\"][\"std\"]*100:.2f}% | params: {m[\"trainable_params_m\"]}M ({m[\"trainable_param_pct\"]}%) | time: {m[\"training_time_hours\"]}h | ckpt: {m[\"checkpoint_size_mb\"]}MB')
" 2>/dev/null || echo "  (comparison not ready)"
    fi
}

# ============================================================
# Main
# ============================================================
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_VISIBLE_DEVICES=0

log "=== Full FT vs LoRA Experiment Started ==="
log "Data: $DATA_ROOT"
log "Cache: $CACHE_DIR"
log "Output: $OUTPUT_DIR"
log ""

run_phase prereqs   && phase_prereqs
run_phase fullft_train && phase_fullft_train
run_phase lora_train   && { phase_lora_train 4 "lora_r4"; phase_lora_train 16 "lora_r16"; }
run_phase infer        && phase_infer
run_phase eval         && phase_eval
run_phase aggregate    && phase_aggregate
phase_summary

log "=== Experiment Finished ==="
