# Roadmap Hợp Nhất (2 phương pháp)

> Tổng hợp hai hướng triển khai: (1) Hybrid CV + CNN, (2) Full End-to-End YOLO + U-Net. Chọn một hướng hoặc chạy song song.

---

## Phương pháp 1: Hybrid CV + CNN (Nhanh, ít dữ liệu)

### G0: Setup
```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python albumentations numpy
```

### G1: Tiền xử lý (kế thừa `hybrid/hybrid.py`)
```python
# preprocess_hybrid.py
import cv2, numpy as np

def preprocess_hybrid(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    gray = clahe.apply(gray)
    blured = cv2.bilateralFilter(gray, 0, 75, 75)
    adap = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, otsu = cv2.threshold(adap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img3 = cv2.merge([otsu, otsu, otsu])  # 3 kênh
    return img3
```

### G2: Dataset + DataLoader
```python
# dataset_hybrid.py
import cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from preprocess_hybrid import preprocess_hybrid

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

class Cv2PreprocessDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_bgr = cv2.imread(path)
        img_bgr = preprocess_hybrid(img_bgr)
        # BGR -> RGB PIL
        img_pil = transforms.functional.to_pil_image(img_bgr[:, :, ::-1])
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, label

# Khi có dữ liệu thật:
# full_ds = Cv2PreprocessDataset('data/train', transform)
# Nếu chưa có, tạm dùng FakeData minh họa
```

### G3: Train/Val CNN
```python
# train_cnn_hybrid.py
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from dataset_hybrid import Cv2PreprocessDataset, transform

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# full_ds = Cv2PreprocessDataset('data/train', transform)
full_ds = datasets.FakeData(transform=transform)  # placeholder
train_len = int(0.8*len(full_ds)); val_len = len(full_ds) - train_len
train_ds, val_ds = random_split(full_ds, [train_len, val_len])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = SmallCNN(num_classes=10).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def eval_step(loader):
    model.eval(); tot_loss=tot_acc=0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            tot_loss += loss_fn(preds, labels).item()*len(imgs)
            tot_acc  += (preds.argmax(1)==labels).sum().item()
    return tot_loss/len(loader.dataset), tot_acc/len(loader.dataset)

for epoch in range(5):
    model.train(); tot_train=0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        loss = loss_fn(model(imgs), labels)
        opt.zero_grad(); loss.backward(); opt.step()
        tot_train += loss.item()*len(imgs)
    train_loss = tot_train/len(train_loader.dataset)
    val_loss, val_acc = eval_step(val_loader)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

torch.save(model.state_dict(), 'runs/cnn_hybrid.pt')
```

**Khi nào dùng?**
- Dữ liệu ít, muốn baseline nhanh.
- Kết hợp đặc trưng hình học (contour) và CNN nhẹ.

---

## Phương pháp 2: Full End-to-End YOLO + U-Net (Đầy đủ pipeline)

### G0: Setup
```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics segmentation-models-pytorch opencv-python albumentations tensorboard
```

### G1: Dataset
```powershell
New-Item -ItemType Directory -Force -Path data\train\images,data\train\labels,data\train\masks,data\val\images,data\val\labels,data\val\masks
@"
path: C:/project/picture-hust/data
train: train/images
val: val/images
nc: 1
names: ['defect']
"@ | Out-File -Encoding utf8 data.yaml
pip install labelImg
labelImg data\train\images data\train\labels
```

### G2: Train YOLO (detection)
```python
# train_yolo.py
import torch
from ultralytics import YOLO

device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16, device=device,
            project='runs/detect', name='xray_yolo')
metrics = model.val()
print('mAP@0.5:', metrics.box.map50)
```

### G3: Train U-Net (segmentation)
```python
# dataset.py
import torch, cv2, numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

class XraySegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms):
        self.img_files = sorted(Path(img_dir).glob('*.png'))
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms
    def __len__(self): return len(self.img_files)
    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_files[idx]), cv2.IMREAD_GRAYSCALE)
        mask_path = self.mask_dir / (self.img_files[idx].stem + '.png')
        mask = cv2.imread(str(mask_path), 0) if mask_path.exists() else np.zeros_like(img)
        mask = (mask > 127).astype(np.float32)
        aug = self.transforms(image=img, mask=mask)
        return aug['image'], aug['mask']

train_transform = A.Compose([
    A.Resize(512,512), A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=0.0, std=1.0), ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(512,512), A.Normalize(mean=0.0, std=1.0), ToTensorV2(),
])
```

```python
# train_unet.py
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import XraySegDataset, train_transform, val_transform

train_ds = XraySegDataset('data/train/images', 'data/train/masks', train_transform)
val_ds   = XraySegDataset('data/val/images',   'data/val/masks',   val_transform)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, classes=1).to(device)
loss_fn = smp.losses.DiceLoss(mode='binary')
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

def dice(pred, target, thr=0.5):
    pred = (pred > thr).float(); inter = (pred*target).sum()
    return (2*inter)/(pred.sum()+target.sum()+1e-8)

best = 0
for epoch in range(50):
    model.train();
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        loss = loss_fn(model(imgs), masks)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); v_dice = v_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            v_loss += loss_fn(preds, masks).item()
            v_dice += dice(torch.sigmoid(preds), masks).item()
    v_loss /= max(1, len(val_loader)); v_dice /= max(1, len(val_loader))
    if v_dice > best:
        best = v_dice; torch.save(model.state_dict(), 'runs/unet/best.pt')
    print(f"Epoch {epoch+1}: val_dice={v_dice:.4f}, best={best:.4f}")
```

### G4: Pipeline YOLO -> U-Net
```python
# pipeline.py
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import torch, cv2, json, numpy as np

yolo = YOLO('runs/detect/xray_yolo/weights/best.pt')
unet = smp.Unet(encoder_name='resnet34', in_channels=1, classes=1).cuda()
unet.load_state_dict(torch.load('runs/unet/best.pt'))
unet.eval()

def process(img_path, out_img='result.png', out_json='result.json'):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    res = yolo(img_path)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    final_mask = np.zeros((h,w), dtype=np.uint8)
    for x1,y1,x2,y2 in boxes:
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        roi = gray[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (512,512))
        roi_t = torch.from_numpy(roi_resized).float().unsqueeze(0).unsqueeze(0).cuda()/255
        with torch.no_grad():
            mask = torch.sigmoid(unet(roi_t)).cpu().numpy()[0,0]
        mask = cv2.resize(mask, (x2-x1, y2-y1))
        mask_bin = (mask>0.5).astype(np.uint8)*255
        final_mask[y1:y2, x1:x2] = np.maximum(final_mask[y1:y2, x1:x2], mask_bin)
        cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
    img[final_mask>0] = [0,0,255]
    defect_ratio = float((final_mask>0).sum() / (h*w) * 100)
    cv2.imwrite(out_img, img)
    with open(out_json,'w') as f:
        json.dump({'boxes': len(boxes), 'defect_ratio': defect_ratio}, f, indent=2)
    return {'boxes': len(boxes), 'defect_ratio': defect_ratio}

print(process('data/val/images/test.png'))
```

### G5: Báo cáo & Checklist
- Mục tiêu: mAP@0.5 ≥ 0.70 (YOLO), Dice ≥ 0.75 (U-Net), FPS ~40 trên GPU.
- Bảng so sánh: YOLOv8n/YOLOv11/RT-DETR; U-Net/DeepLabV3+; Pipeline FPS/size.
- Xuất ONNX nếu cần: `model.export(format='onnx')` với YOLO; `torch.onnx.export` cho U-Net.

---

## Chọn hướng nào?
- **Hybrid CV + CNN**: nhanh, ít dữ liệu, dễ debug; phù hợp baseline/POC.
- **Full YOLO + U-Net**: mạnh hơn, end-to-end, cần labeling bbox + mask, training lâu hơn nhưng cho kết quả tốt hơn cho ĐATN.
