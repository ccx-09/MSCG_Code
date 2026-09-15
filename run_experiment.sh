#!/bin/bash
set -eo pipefail

# ============================================================
# MSCG 5-Fold Cross-Validation Experiment
# SAM+LoRA / DeepLabV3+ / XGBoost
#
# Usage:
#   bash run_experiment.sh                           # 跑全部（5折完整实验）
#   bash run_experiment.sh --phase sam               # 只跑 SAM 全部5折
#   bash run_experiment.sh --phase sam --fold 0      # 只跑 SAM 的 fold 0
#   bash run_experiment.sh --phase deeplabv3 --fold 0
#   bash run_experiment.sh --phase xgb --fold 0
#   bash run_experiment.sh --phase eval              # 全部5折评估+对比
#
# Optional path variables (defaults are relative to this script):
#   PROJECT_ROOT, DATA_ROOT, SAM_CKPT, CACHE_DIR, OUTPUT_DIR
# Example:
#   DATA_ROOT=/path/to/MSCG_dataset \
#   SAM_CKPT=/path/to/sam_vit_h_4b8939.pth \
#   bash run_experiment.sh --phase eval
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
METHODS_DIR="${METHODS_DIR:-$PROJECT_ROOT/methods}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$PROJECT_ROOT/analysis}"
DATA_ROOT="${DATA_ROOT:-${MSCG_DATA_ROOT:-}}"
SAM_CKPT="${SAM_CKPT:-${SAM_CHECKPOINT:-}}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/cache/mscg}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/artifacts/newdata/ws1_kfold}"
LOG_DIR="$CACHE_DIR/logs"
FOLDS="${FOLDS:-5}"

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/experiment.log"; }
section() { echo "" | tee -a "$LOG_DIR/experiment.log"; log "========== $1 =========="; }

# --- Parse arguments ---
RUN_ALL=true
SPECIFIC_PHASE=""
SPECIFIC_FOLD=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)  RUN_ALL=false; SPECIFIC_PHASE="$2"; shift 2 ;;
        --fold)   SPECIFIC_FOLD="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$SPECIFIC_FOLD" ]; then
    FOLD_RANGE="$SPECIFIC_FOLD"
    log "Running fold $SPECIFIC_FOLD only"
else
    FOLD_RANGE=$(seq 0 $((FOLDS - 1)))
fi
run_phase() {
    if $RUN_ALL || [ "$SPECIFIC_PHASE" == "$1" ]; then
        return 0
    else
        return 1
    fi
}

# ============================================================
phase_prereqs() {
    section "Phase 0: Prerequisites Check"

    local errors=0

    if [ ! -d "$DATA_ROOT/composites" ]; then
        echo "  ❌ Data not found: $DATA_ROOT"
        errors=$((errors + 1))
    else
        local img_count=$(find "$DATA_ROOT/composites" -name "*.jpg" | wc -l)
        echo "  ✅ Dataset: $img_count images"
    fi

    if [ -f "$SAM_CKPT" ]; then
        echo "  ✅ SAM checkpoint: $(du -h "$SAM_CKPT" | cut -f1)"
    else
        echo "  ❌ SAM checkpoint not found: $SAM_CKPT"
        errors=$((errors + 1))
    fi

    if python3 -c "import torch; print(f'  ✅ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')" 2>/dev/null; then
        :
    else
        echo "  ❌ CUDA not available"
        errors=$((errors + 1))
    fi

    for pkg in torch numpy PIL cv2 albumentations sklearn xgboost tqdm segmentation_models_pytorch; do
        if python3 -c "import $pkg" 2>/dev/null; then
            :
        else
            echo "  ❌ Package missing: $pkg"
            errors=$((errors + 1))
        fi
    done

    for fold in $FOLD_RANGE; do
        for split in train val test; do
            if [ ! -f "$DATA_ROOT/folds/fold${fold}_${split}.json" ]; then
                echo "  ❌ Missing fold JSON: fold${fold}_${split}.json"
                errors=$((errors + 1))
            fi
        done
    done
    echo "  ✅ All fold JSONs present"

    if [ $errors -gt 0 ]; then
        echo "  ❌ $errors error(s) found. Fix before running."
        exit 1
    fi
    echo "  ✅ All checks passed!"
}

# ============================================================
phase_sam_train() {
    section "Phase 1: SAM+LoRA Training (5 folds)"

    for fold in $FOLD_RANGE; do
        if [ -f "$CACHE_DIR/sam/fold$fold/best_model.pt" ]; then
            log "  ⏩ SAM fold $fold already done, skipping"
            continue
        fi
        log "  ▶ SAM fold $fold training..."
        python "$METHODS_DIR/train_fold_sam.py" \
            --fold $fold \
            --ws1_root "$DATA_ROOT" \
            --sam_checkpoint "$SAM_CKPT" \
            --policy P1 --epochs 50 \
            --batch_size 4 --gradient_accumulation_steps 4 \
            --lr 5e-5 --early_stopping_patience 3 \
            --output_dir "$CACHE_DIR/sam/fold$fold" \
            2>&1 | tee -a "$LOG_DIR/sam_fold${fold}.log"
        log "  ✅ SAM fold $fold completed"
    done
}

# ============================================================
phase_deeplabv3_train() {
    section "Phase 2: DeepLabV3+ Training (5 folds)"

    for fold in $FOLD_RANGE; do
        if [ -f "$CACHE_DIR/deeplabv3/fold$fold/best_model.pth" ]; then
            log "  ⏩ DeepLabV3+ fold $fold already done, skipping"
            continue
        fi
        log "  ▶ DeepLabV3+ fold $fold training..."
        python "$METHODS_DIR/train_deeplabv3_fixed.py" \
            --fold $fold \
            --data_root "$DATA_ROOT" \
            --train_json "$DATA_ROOT/folds/fold${fold}_train.json" \
            --val_json "$DATA_ROOT/folds/fold${fold}_val.json" \
            --output "$CACHE_DIR/deeplabv3" \
            2>&1 | tee -a "$LOG_DIR/deeplabv3_fold${fold}.log"
        log "  ✅ DeepLabV3+ fold $fold completed"
    done
}

# ============================================================
phase_xgb_extract() {
    section "Phase 3a: XGBoost Feature Extraction"

    for fold in $FOLD_RANGE; do
        for split in train val test; do
            local feat_file="$CACHE_DIR/xgb_features/fold${fold}_${split}_features.npz"
            if [ -f "$feat_file" ]; then
                log "  ⏩ Features fold $fold/$split already done, skipping"
                continue
            fi
            log "  ▶ Extracting features fold $fold/$split..."
            python "$METHODS_DIR/extract_fold_features.py" \
                --fold_manifest "$DATA_ROOT/folds/fold${fold}_${split}.json" \
                --ws1_root "$DATA_ROOT" \
                --output_dir "$CACHE_DIR/xgb_features" \
                2>&1 | tee -a "$LOG_DIR/xgb_extract_fold${fold}_${split}.log"
            log "  ✅ Features fold $fold/$split done"
        done
    done
}

# ============================================================
phase_xgb_train() {
    section "Phase 3b: XGBoost Training (5 folds)"

    for fold in $FOLD_RANGE; do
        if [ -f "$CACHE_DIR/xgb_models/fold${fold}_model.json" ]; then
            log "  ⏩ XGBoost fold $fold already done, skipping"
            continue
        fi
        log "  ▶ XGBoost fold $fold training..."
        python "$METHODS_DIR/train_fold_xgb.py" \
            --features_dir "$CACHE_DIR/xgb_features" \
            --config "$METHODS_DIR/xgb_minimal.yaml" \
            --output_dir "$CACHE_DIR/xgb_models" \
            --fold $fold \
            2>&1 | tee -a "$LOG_DIR/xgb_train_fold${fold}.log"
        log "  ✅ XGBoost fold $fold completed"
    done
}

# ============================================================
phase_infer() {
    section "Phase 4: Inference"

    for fold in $FOLD_RANGE; do
        local test_json="$DATA_ROOT/folds/fold${fold}_test.json"

        # --- SAM Inference ---
        local sam_model="$CACHE_DIR/sam/fold$fold/best_model.pt"
        if [ ! -f "$sam_model" ]; then
            log "  ⏩ SAM model not found (fold $fold), skipping infer"
        elif [ -f "$CACHE_DIR/pred_sam/fold$fold/inference_manifest.json" ]; then
            log "  ⏩ SAM infer fold $fold already done, skipping"
        else
            log "  ▶ SAM infer fold $fold..."
            mkdir -p "$CACHE_DIR/pred_sam/fold$fold"
            python "$METHODS_DIR/infer_fold_sam.py" \
                --fold $fold \
                --model_dir "$CACHE_DIR/sam/fold$fold" \
                --ws1_root "$DATA_ROOT" \
                --output_dir "$CACHE_DIR/pred_sam/fold$fold" \
                --policy P1 \
                2>&1 | tee -a "$LOG_DIR/infer_sam_fold${fold}.log"
            log "  ✅ SAM infer fold $fold done"
        fi

        # --- DeepLabV3+ Inference ---
        local dl_model="$CACHE_DIR/deeplabv3/fold$fold/best_model.pth"
        if [ ! -f "$dl_model" ]; then
            log "  ⏩ DeepLabV3+ model not found (fold $fold), skipping infer"
        elif [ -d "$CACHE_DIR/pred_deeplabv3/fold$fold" ] && [ "$(ls -A "$CACHE_DIR/pred_deeplabv3/fold$fold" 2>/dev/null)" ]; then
            log "  ⏩ DeepLabV3+ infer fold $fold already done, skipping"
        else
            log "  ▶ DeepLabV3+ infer fold $fold..."
            mkdir -p "$CACHE_DIR/pred_deeplabv3/fold$fold"
            python "$METHODS_DIR/infer_deeplabv3.py" \
                --test_json "$test_json" \
                --data_root "$DATA_ROOT" \
                --model "$dl_model" \
                --output "$CACHE_DIR/pred_deeplabv3/fold$fold" \
                --fold $fold \
                2>&1 | tee -a "$LOG_DIR/infer_deeplabv3_fold${fold}.log"
            log "  ✅ DeepLabV3+ infer fold $fold done"
        fi

        # --- XGBoost Inference ---
        local xgb_model="$CACHE_DIR/xgb_models/fold${fold}_model.json"
        if [ ! -f "$xgb_model" ]; then
            log "  ⏩ XGBoost model not found (fold $fold), skipping infer"
        elif [ -d "$CACHE_DIR/pred_xgb/fold$fold" ] && [ "$(ls -A "$CACHE_DIR/pred_xgb/fold$fold" 2>/dev/null)" ]; then
            log "  ⏩ XGBoost infer fold $fold already done, skipping"
        else
            log "  ▶ XGBoost infer fold $fold..."
            mkdir -p "$CACHE_DIR/pred_xgb/fold$fold"
            python "$METHODS_DIR/infer_fold_predictions.py" \
                --model_dir "$CACHE_DIR/xgb_models" \
                --features_dir "$CACHE_DIR/xgb_features" \
                --fold_manifest "$test_json" \
                --ws1_root "$DATA_ROOT" \
                --output_dir "$CACHE_DIR/pred_xgb/fold$fold" \
                --fold $fold \
                2>&1 | tee -a "$LOG_DIR/infer_xgb_fold${fold}.log"
            log "  ✅ XGBoost infer fold $fold done"
        fi
    done
}

# ============================================================
phase_eval() {
    section "Phase 5: Granular Evaluation"

    for fold in $FOLD_RANGE; do
        local test_json="$DATA_ROOT/folds/fold${fold}_test.json"

        for method in sam deeplabv3 xgb; do
            local pred_dir="$CACHE_DIR/pred_${method}/fold$fold"
            local out_dir="$OUTPUT_DIR/fold$fold/$method"
            if [ -f "$out_dir/summary.json" ]; then
                log "  ⏩ Eval $method fold $fold already done, skipping"
                continue
            fi
            if [ ! -d "$pred_dir" ] || [ -z "$(ls -A "$pred_dir" 2>/dev/null)" ]; then
                log "  ⚠ No predictions for $method fold $fold, skipping eval"
                continue
            fi
            log "  ▶ Evaluating $method fold $fold..."
            mkdir -p "$out_dir"
            python "$ANALYSIS_DIR/ws1_granular_eval.py" \
                --ws1-json "$test_json" \
                --pred-dir "$pred_dir" \
                --gt-dir "$DATA_ROOT/masks" \
                --out-dir "$out_dir" \
                2>&1 | tee -a "$LOG_DIR/eval_${method}_fold${fold}.log"
            log "  ✅ Eval $method fold $fold done"
        done
    done
}

# ============================================================
phase_compare() {
    section "Phase 6: Three-Way Comparison"

    if [ -f "$OUTPUT_DIR/three_way_comparison.json" ]; then
        log "  ⏩ Comparison already done, skipping"
    else
        log "  ▶ Running three-way comparison..."
        python "$ANALYSIS_DIR/compare_all_methods_ws1.py" \
            --stats_dir "$OUTPUT_DIR" \
            --output "$OUTPUT_DIR/three_way_comparison.json" \
            2>&1 | tee -a "$LOG_DIR/comparison.log"
        log "  ✅ Comparison done → $OUTPUT_DIR/three_way_comparison.json"
    fi
}

# ============================================================
phase_summary() {
    section "Experiment Summary"
    echo "  Cache:     $CACHE_DIR"
    echo "  Output:    $OUTPUT_DIR"
    echo "  Logs:      $LOG_DIR"
    echo ""

    echo "  === Training Status ==="
    for fold in $FOLD_RANGE; do
        local sam_ckpt="$CACHE_DIR/sam/fold$fold/best_model.pt"
        local dl_ckpt="$CACHE_DIR/deeplabv3/fold$fold/best_model.pth"
        local xgb_ckpt="$CACHE_DIR/xgb_models/fold${fold}_model.json"
        local sam_s=$( [ -f "$sam_ckpt" ] && echo "✅" || echo "⏳" )
        local dl_s=$(  [ -f "$dl_ckpt" ]  && echo "✅" || echo "⏳" )
        local xgb_s=$( [ -f "$xgb_ckpt" ] && echo "✅" || echo "⏳" )
        echo "  Fold $fold: SAM $sam_s | DeepLabV3+ $dl_s | XGBoost $xgb_s"
    done

    if [ -f "$OUTPUT_DIR/three_way_comparison.json" ]; then
        echo ""
        python3 -c "
import json
d = json.load(open('$OUTPUT_DIR/three_way_comparison.json'))
methods = d.get('methods', [])
comp = d.get('comparison', {})
for m in methods:
    s = comp.get(m, {}).get('miou', {})
    if s:
        print(f'  {m}: mIoU = {s[\"mean\"]:.2f}% ± {s[\"std\"]:.2f}%')
print(f'  Best: {d.get(\"summary\", {}).get(\"best_method\", \"?\")}')
"
    fi
}

# ============================================================
# Main
# ============================================================
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

log "=== MSCG Experiment Started ==="
log "Data: $DATA_ROOT"
log "Cache: $CACHE_DIR"
log "Output: $OUTPUT_DIR"
log ""

run_phase prereqs   && phase_prereqs
run_phase sam       && phase_sam_train
run_phase deeplabv3 && phase_deeplabv3_train
run_phase xgb       && { phase_xgb_extract; phase_xgb_train; }
run_phase infer     && phase_infer
run_phase eval      && phase_eval
run_phase compare   && phase_compare
phase_summary

log "=== Experiment Finished ==="
