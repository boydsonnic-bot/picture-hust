#!/usr/bin/env python3
"""
Lightweight inference pipeline: YOLOv8 detection -> crop ROI -> U-Net ONNX segmentation
Post-process masks, compute defect ratio and save visualization.

Usage:
  python scripts/inference_pipeline.py --image test/demo_image.png \
      --yolo best.pt --unet unet.onnx --out results/output.png

Notes:
 - YOLO model: prefer `ultralytics.YOLO` (pt or model name). If you only have an ONNX
   YOLO export, you can modify the `run_yolo_onnx` stub.
 - U-Net: expected as ONNX model taking input shape [1, C, H, W] (C=1 or 3).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


def load_yolo(model_path: str):
    if YOLO is None:
        raise RuntimeError('ultralytics package is required for YOLO usage (pip install ultralytics)')
    return YOLO(model_path)


def run_yolo(yolo_model, image: np.ndarray, conf: float = 0.25):
    # ultralytics YOLO inference
    results = yolo_model(image)
    r = results[0]
    boxes = []
    scores = []
    # safer extraction
    for b in r.boxes:
        try:
            xyxy = b.xyxy.cpu().numpy().astype(int).flatten().tolist()
            conf_score = float(b.conf.cpu().numpy())
        except Exception:
            # fallback: attributes might be numpy already
            xyxy = np.array(b.xyxy).astype(int).flatten().tolist()
            conf_score = float(b.conf)
        boxes.append(xyxy)
        scores.append(conf_score)
    return np.array(boxes, dtype=int), np.array(scores)


def load_unet_onnx(path: str):
    if ort is None:
        raise RuntimeError('onnxruntime is required for UNet ONNX inference (pip install onnxruntime)')
    return ort.InferenceSession(path, providers=['CPUExecutionProvider'])


def run_unet_onnx(session, roi: np.ndarray, in_ch=1, input_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    # roi: HxW or HxWx3
    h, w = input_size
    if roi.ndim == 3 and roi.shape[2] == 3 and in_ch == 1:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if roi.ndim == 2:
        inp = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
        inp = inp.astype(np.float32) / 255.0
        inp = inp[np.newaxis, np.newaxis, :, :]
    else:
        inp = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
        inp = inp.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...]

    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: inp})[0]
    # assume output is [1,1,H,W] or [1,H,W]
    mask = out[0]
    if mask.ndim == 3:
        # [C,H,W]
        mask = mask[0]
    mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def postprocess_mask(mask: np.ndarray, thresh: float = 0.5, min_area: int = 50) -> np.ndarray:
    bw = (mask >= thresh).astype(np.uint8) * 255
    # morphology to close holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(bw)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out // 255


def visualize(image: np.ndarray, boxes: List[List[int]], masks: List[np.ndarray], scores: List[float], out_path: str):
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    for (box, mask, score) in zip(boxes, masks, scores):
        x1, y1, x2, y2 = box
        color = (0, 0, 255)
        # draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f'{score:.2f}', (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # overlay mask
        if mask is not None:
            if mask.ndim == 2:
                colored = np.zeros_like(vis)
                colored[y1:y2, x1:x2, 2] = (mask * 255).astype(np.uint8)
                cv2.addWeighted(colored, 0.5, overlay, 0.5, 0, overlay)
    out = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)
    os.makedirs(Path(out_path).parent, exist_ok=True)
    cv2.imwrite(out_path, out)
    return out


def compute_defect_ratio(masks: List[np.ndarray], boxes: List[List[int]]) -> float:
    total_defect = 0
    total_pixels = 0
    for mask, box in zip(masks, boxes):
        x1, y1, x2, y2 = box
        area_box = (x2 - x1) * (y2 - y1)
        total_defect += int((mask > 0).sum())
        total_pixels += area_box
    return (total_defect / total_pixels * 100.0) if total_pixels > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--yolo', required=True, help='YOLO model path or name (ultralytics)')
    p.add_argument('--unet', required=True, help='U-Net ONNX path')
    p.add_argument('--out', default='results/out.png')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--mask-thresh', type=float, default=0.5)
    p.add_argument('--min-area', type=int, default=30)
    args = p.parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')

    yolo_model = load_yolo(args.yolo)
    boxes, scores = run_yolo(yolo_model, img, conf=args.conf)

    unet_session = load_unet_onnx(args.unet)

    masks = []
    good_boxes = []
    good_scores = []
    for (box, score) in zip(boxes, scores):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = img[y1:y2, x1:x2]
        mask_pred = run_unet_onnx(unet_session, roi, in_ch=1 if roi.ndim==2 or roi.shape[2]==1 else 3)
        mask_pp = postprocess_mask(mask_pred, thresh=args.mask_thresh, min_area=args.min_area)
        masks.append(mask_pp)
        good_boxes.append([x1, y1, x2, y2])
        good_scores.append(float(score))

    defect_ratio = compute_defect_ratio(masks, good_boxes)

    vis = visualize(img, good_boxes, masks, good_scores, args.out)

    out = {
        'num_defects': len(good_boxes),
        'defect_ratio_percent': defect_ratio,
        'boxes': good_boxes,
        'scores': good_scores,
        'visualization': str(Path(args.out).resolve())
    }

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
