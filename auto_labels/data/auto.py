import os
from ultralytics import YOLO

model_path = r'C:\project\picture-hust\auto_labels\data\best.pt'
images_dir = r'C:\project\picture-hust\auto_labels\data\val\images'
labels_dir = r'C:\project\picture-hust\auto_labels\data\val\labels'

os.makedirs(labels_dir, exist_ok=True)

CONF_THRES = 0.15
IOU_THRES = 0.5

model = YOLO(model_path)

results = model.predict(
    source=images_dir,
    conf=CONF_THRES,
    iou=IOU_THRES,
    save=False,
    stream=True
)

no_detect = 0
has_detect = 0

for result in results:
    img_name = os.path.splitext(os.path.basename(result.path))[0]

    # ❗ nếu KHÔNG detect → bỏ qua, KHÔNG tạo txt
    if result.boxes is None or len(result.boxes) == 0:
        no_detect += 1
        continue

    has_detect += 1
    label_path = os.path.join(labels_dir, f"{img_name}.txt")

    img_h, img_w = result.orig_shape

    with open(label_path, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls[0])

            # xywh pixel
            x, y, w, h = box.xywh[0].tolist()

            # normalize
            x /= img_w
            y /= img_h
            w /= img_w
            h /= img_h

            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

print("✅ Auto labeling completed")
print(f"📦 Images with labels: {has_detect}")
print(f"🚫 Images with NO detect: {no_detect}")
