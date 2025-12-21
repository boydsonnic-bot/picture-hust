import os
import cv2
import numpy as np
from tqdm import tqdm

# ====== PATH ======
IMAGES_DIR = r"C:\\project\\picture-hust\\auto_labels\\data\\val\\images"
LABELS_DIR = r"C:\\project\\picture-hust\\auto_labels\\data\\val\\labels"
MASKS_DIR  = r"C:\\project\\picture-hust\\auto_labels\\data\\val\\masks"

os.makedirs(MASKS_DIR, exist_ok=True)

# ====== MAIN ======
for img_name in tqdm(os.listdir(IMAGES_DIR)):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMAGES_DIR, img_name)
    base_name, _ = os.path.splitext(img_name)
    label_path = os.path.join(LABELS_DIR, base_name + ".txt")
    mask_path = os.path.join(MASKS_DIR, base_name + ".png")

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # mask đen
    mask = np.zeros((h, w), dtype=np.uint8)

    if not os.path.exists(label_path):
        cv2.imwrite(mask_path, mask)
        continue

    with open(label_path, "r") as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())

            # YOLO → pixel
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    cv2.imwrite(mask_path, mask)

print("✅ Convert label → mask DONE")
