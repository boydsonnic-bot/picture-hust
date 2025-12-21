import os
import random
import shutil
from tqdm import tqdm

SRC_IMAGES = r"C:\project\picture-hust\auto_labels\data\val\images"
SRC_MASKS  = r"C:\project\picture-hust\auto_labels\data\val\masks"
SRC_LABELS = r"C:\project\picture-hust\auto_labels\data\val\labels"  # YOLO txt để xác định OK/NG

DST_ROOT   = r"C:\project\picture-hust\FULL\data-unet"

TRAIN_IMG = os.path.join(DST_ROOT, "train/images")
TRAIN_MSK = os.path.join(DST_ROOT, "train/masks")
VAL_IMG   = os.path.join(DST_ROOT, "val/images")
VAL_MSK   = os.path.join(DST_ROOT, "val/masks")

for p in [TRAIN_IMG, TRAIN_MSK, VAL_IMG, VAL_MSK]:
    os.makedirs(p, exist_ok=True)

images = [f for f in os.listdir(SRC_IMAGES) if f.lower().endswith((".jpg", ".png"))]

def get_class(fname: str) -> str:
    base, _ = os.path.splitext(fname)
    label_path = os.path.join(SRC_LABELS, base + ".txt")
    # Nếu có label và có ít nhất 1 dòng bbox → xem là NG, ngược lại OK
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return "NG"
    return "OK"

# Phân lớp trước để chia giữ tỉ lệ OK/NG
ok_files, ng_files = [], []
for f in images:
    (ng_files if get_class(f) == "NG" else ok_files).append(f)

random.shuffle(ok_files)
random.shuffle(ng_files)

def stratified_split(file_list, ratio=0.8):
    split_idx = int(ratio * len(file_list))
    return file_list[:split_idx], file_list[split_idx:]

ok_train, ok_val = stratified_split(ok_files)
ng_train, ng_val = stratified_split(ng_files)

train_files = ok_train + ng_train
val_files   = ok_val + ng_val

random.shuffle(train_files)
random.shuffle(val_files)

def copy_split(file_list, dst_img, dst_msk, desc):
    for f in tqdm(file_list, desc=desc):
        base, _ = os.path.splitext(f)
        img_src = os.path.join(SRC_IMAGES, f)
        msk_src = os.path.join(SRC_MASKS, base + ".png")  # mask luôn .png
        if not os.path.exists(msk_src):
            continue  # hoặc raise/print cảnh báo
        shutil.copy(img_src, os.path.join(dst_img, f))
        shutil.copy(msk_src, os.path.join(dst_msk, base + ".png"))

copy_split(train_files, TRAIN_IMG, TRAIN_MSK, "Copy TRAIN")
copy_split(val_files,   VAL_IMG,   VAL_MSK,   "Copy VAL")

print("✅ Split 80–20 DONE")
print(f"Train: {len(train_files)} images")
print(f"Val  : {len(val_files)} images")