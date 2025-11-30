import argparse
import json
import logging
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (list): [x_center, y_center, width, height] (normalized)
        box2 (list): [x_center, y_center, width, height] (normalized)
        
    Returns:
        float: IoU value.
    """
    # Convert to corners: x1, y1, x2, y2
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    
    # Intersection coordinates
    x1 = max(b1_x1, b2_x1)
    y1 = max(b1_y1, b2_y1)
    x2 = min(b1_x2, b2_x2)
    y2 = min(b1_y2, b2_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union = b1_area + b2_area - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_model(model_path, dataset_path, config, logger):
    """
    Evaluates the YOLO model using custom metrics (FROC, NPV).
    """
    logger.info(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    val_images_dir = dataset_path / "images/val"
    val_labels_dir = dataset_path / "labels/val"
    
    image_files = list(val_images_dir.glob("*.png"))
    if not image_files:
        logger.error(f"No validation images found in {val_images_dir}")
        return

    logger.info(f"Found {len(image_files)} validation images.")
    
    # Storage for all predictions and ground truths
    # Structure: List of dicts per image
    results = []
    
    # FROC parameters
    iou_threshold = 0.1 # Loose threshold for small objects
    
    for img_path in tqdm(image_files, desc="Running Inference"):
        if img_path.name.startswith('.'):
            continue
            
        # 1. Load Ground Truth
        label_path = val_labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    # Only care about class 0 (MSX Plasmodia) for now? 
                    # Or all classes? Let's assume class 0 is the target.
                    if int(cls) == 0: 
                        gt_boxes.append([x, y, w, h])
        
        # 2. Run Inference
        # Run with low confidence to capture the full curve
        preds = model.predict(img_path, conf=0.01, verbose=False)[0]
        
        pred_boxes = []
        pred_scores = []
        
        for box in preds.boxes:
            if int(box.cls) == 0: # Filter for target class
                # box.xywhn returns [x, y, w, h] normalized
                xywhn = box.xywhn[0].tolist()
                conf = float(box.conf)
                pred_boxes.append(xywhn)
                pred_scores.append(conf)
                
        results.append({
            "img_path": str(img_path),
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores
        })

    logger.info("Inference complete. Calculating metrics...")
    
    # --- Calculate FROC and NPV ---
    # We need to sweep thresholds
    thresholds = np.linspace(0.01, 0.95, 50)
    
    froc_data = [] # List of (FP_per_mm2, Sensitivity)
    npv_data = [] # List of (Threshold, NPV)
    
    # Image area in mm2 (approximate)
    # Patch size 640px. At 40x, 1px ~ 0.25um? 
    # Let's assume 1px = 0.5um for now (standard 20x). 
    # 640px * 0.5um = 320um = 0.32mm.
    # Area = 0.32 * 0.32 = 0.1024 mm2 per patch.
    mm2_per_patch = 0.1024 
    total_area_mm2 = len(image_files) * mm2_per_patch
    
    for thresh in tqdm(thresholds, desc="Sweeping Thresholds"):
        tp_total = 0
        fp_total = 0
        fn_total = 0
        tn_total = 0 # True Negatives (patches with no GT and no Pred)
        
        for res in results:
            gt = res["gt_boxes"]
            # Filter predictions by threshold
            preds = [p for p, s in zip(res["pred_boxes"], res["pred_scores"]) if s >= thresh]
            
            # Match predictions to GT
            # Greedy matching
            matched_gt = set()
            fp_count = 0
            
            for p_box in preds:
                best_iou = 0
                best_gt_idx = -1
                
                for i, g_box in enumerate(gt):
                    if i in matched_gt:
                        continue
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= iou_threshold:
                    matched_gt.add(best_gt_idx)
                else:
                    fp_count += 1
            
            tp = len(matched_gt)
            fn = len(gt) - tp
            fp = fp_count
            
            tp_total += tp
            fp_total += fp
            fn_total += fn
            
            # Patch-level NPV calculation logic:
            # A "Negative" patch is one with NO predictions.
            # A "True Negative" patch is one with NO predictions AND NO GT.
            # A "False Negative" patch is one with NO predictions BUT HAS GT.
            if len(preds) == 0:
                if len(gt) == 0:
                    tn_total += 1
                else:
                    # This patch was predicted negative (clean), but had parasites!
                    # This contributes to the "False Negative" count for NPV
                    pass 
        
        # FROC Metrics
        sensitivity = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        fp_per_mm2 = fp_total / total_area_mm2
        froc_data.append((fp_per_mm2, sensitivity))
        
        # NPV Metrics (Patch Level)
        # NPV = TN / (TN + FN_patches)
        # We need to count FN patches specifically
        fn_patches = 0
        tn_patches = 0
        for res in results:
            gt = res["gt_boxes"]
            preds = [p for p, s in zip(res["pred_boxes"], res["pred_scores"]) if s >= thresh]
            
            if len(preds) == 0:
                if len(gt) == 0:
                    tn_patches += 1
                else:
                    fn_patches += 1
        
        npv = tn_patches / (tn_patches + fn_patches) if (tn_patches + fn_patches) > 0 else 0
        npv_data.append((thresh, npv))

    # --- Plotting ---
    # Use the parent of inference_results as the base output directory
    output_dir = Path(config['paths']['inference_results']).parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot FROC
    fps, sens = zip(*froc_data)
    plt.figure(figsize=(10, 6))
    plt.plot(fps, sens, marker='o')
    plt.xlabel('False Positives per mm²')
    plt.ylabel('Sensitivity')
    plt.title('FROC Curve (MSX Detection)')
    plt.grid(True)
    plt.savefig(output_dir / "froc_curve.png")
    plt.close()
    
    # Plot NPV vs Threshold
    threshs, npvs = zip(*npv_data)
    plt.figure(figsize=(10, 6))
    plt.plot(threshs, npvs, marker='x', color='green')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Negative Predictive Value (Patch Level)')
    plt.title('NPV vs Confidence Threshold')
    plt.grid(True)
    plt.savefig(output_dir / "npv_curve.png")
    plt.close()
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    
    # Print Report at specific operating point (e.g. 0.25)
    target_thresh = 0.25
    # Find closest index
    idx = (np.abs(thresholds - target_thresh)).argmin()
    
    print("\n" + "="*40)
    print(f"  Evaluation Report @ Conf={target_thresh:.2f}")
    print("="*40)
    print(f"  Sensitivity:       {sens[idx]:.4f}")
    print(f"  FP per mm²:        {fps[idx]:.4f}")
    print(f"  Patch-Level NPV:   {npvs[idx]:.4f}")
    print("="*40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO with Custom Metrics (FROC, NPV)")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    logger = setup_logging(Path(config['paths']['logs']), 'evaluate_custom')
    
    # Determine model path
    if config['training'].get('use_latest_model', False):
        import importlib
        train_yolo = importlib.import_module("src.main_scripts.02_train_yolo")
        find_latest_best_model = train_yolo.find_latest_best_model
        model_path = find_latest_best_model(Path(config['paths']['model_output_dir']), logger)
    else:
        # Default to the checkpoint defined for inference, as that's likely the "best" known model
        model_path = config['inference'].get('model_checkpoint', None)
        if not model_path:
            # Fallback to training base model (unlikely to be desired but safe)
            model_path = config['training']['yolo_model']
        
    if not model_path:
        logger.error("No model found to evaluate.")
        return

    dataset_path = Path(config['paths']['yolo_dataset'])
    
    evaluate_model(model_path, dataset_path, config, logger)

if __name__ == "__main__":
    main()
