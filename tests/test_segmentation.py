# Segmentasyon kalitesini değerlendirmek icin bu metrikleri kullanabiliriz:
# 1. IoU (Intersection over Union)
# 2. Dice Coefficient
# 3. Pixel Accuracy

import numpy as np
import cv2
import os


def load_mask(path):
    """Maske dosyasını siyah-beyaz (binary) olarak yükler."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (mask > 127).astype(np.uint8)  # 0 veya 1'e çevir


def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union != 0 else 1.0


def compute_dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    return (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) != 0 else 1.0


def compute_pixel_accuracy(pred, gt):
    correct = (pred == gt).sum()
    total = pred.size
    return correct / total

PRED_DIR = "../outputs/masks" 
IMG_DIR = "../data" 

pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith('_transparent.png')]

for fname in pred_files:
    name = fname.replace('_transparent.png', '')
    pred_mask = load_mask(os.path.join(PRED_DIR, fname))
    gt_mask_path = os.path.join(IMG_DIR, f"{name}.png")  # Burada uzantı jpg olarak değiştirildi
    img_path = os.path.join(IMG_DIR, f"{name}.png")

    if not os.path.exists(gt_mask_path):
        print(f"⚠️ Gerçek maske bulunamadı: {gt_mask_path}")
        continue

    if not os.path.exists(img_path):
        print(f"⚠️ Orijinal görsel bulunamadı: {img_path}")
        continue

    gt_mask = load_mask(gt_mask_path)

    iou = compute_iou(pred_mask, gt_mask)
    dice = compute_dice(pred_mask, gt_mask)
    acc = compute_pixel_accuracy(pred_mask, gt_mask)

    print(f"{name} - IoU: {iou:.3f}, Dice: {dice:.3f}, Pixel Accuracy: {acc:.3f}")