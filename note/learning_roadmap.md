# Lộ trình học 8 tuần (từ dễ → khó) cho dự án Computer Vision

**Mục tiêu**: Nâng cấp từ code phát hiện contour cơ bản (`test02.py`) lên hệ thống phát hiện khuyết tật tự động (detection/classification) với Deep Learning.

**Time budget**: 
- Thứ 2-6: 1h/ngày × 5 = 5h/tuần
- Thứ 7-CN: 3h/ngày × 2 = 6h/tuần
- **Tổng: ~10-11h/tuần × 8 tuần = 80-88h**

**Nguyên tắc**: Kiến thức từ dễ → khó; mỗi tuần vừa đọc lý thuyết vừa code thực hành.

---

## 📚 Phân tích code hiện tại (`test02.py`)

**Code bạn đang có**:
```python
gray → GaussianBlur → Otsu threshold → findContours → boundingRect → save
```

**Điểm mạnh**: 
- Xử lý ảnh cơ bản (grayscale, blur, threshold)
- Phát hiện contour và tính area
- CLI arguments, save kết quả

**Hạn chế (cần nâng cấp)**:
- Không có tiền xử lý nâng cao (CLAHE, morphology, adaptive threshold)
- Chưa phân loại (classify) contour là khuyết tật hay nhiễu
- Chưa dùng Deep Learning (CNN) để học feature tự động
- Chưa có detection model (YOLO, Faster R-CNN) để định vị chính xác

---

## 🗓️ Lộ trình 8 tuần (Easy → Hard)

### **TUẦN 1-2: Classical Computer Vision (Nền tảng xử lý ảnh)**

**Mục tiêu**: Nâng cấp preprocessing pipeline (tiền xử lý ảnh tốt hơn)

#### Tuần 1: Adaptive Thresholding & CLAHE
- **Đọc từ PDF** (Chương 1-2 hoặc phần cơ bản): Python basics, NumPy, Matplotlib
- **Đọc thêm (Google search)**:
  - `CLAHE OpenCV` (Contrast Limited Adaptive Histogram Equalization)
  - `Adaptive Threshold vs Otsu`
  - `Morphological operations erosion dilation`
- **Key concepts**:
  - **CLAHE**: Tăng contrast cục bộ (tốt cho ảnh X-ray có độ sáng không đều)
  - **Adaptive Threshold**: Threshold động theo vùng (tốt hơn Otsu khi ảnh có lighting không đồng nhất)
  - **Morphology (erosion/dilation/opening/closing)**: Loại bỏ noise, làm mịn contour

- **Code nâng cấp (3-4h)**:
  ```python
  # test02_v2.py - add CLAHE + adaptive threshold
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  enhanced = clahe.apply(gray)
  adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
  # Morphology
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
  ```

- **Output**: So sánh Otsu vs Adaptive + CLAHE side-by-side

#### Tuần 2: Feature Engineering (Geometric features)
- **Đọc thêm**:
  - `Contour features OpenCV` (area, perimeter, circularity, aspect ratio)
  - `Hu Moments invariant features`
- **Key concepts**:
  - **Geometric features**: Area, perimeter, circularity = 4π×area/perimeter², aspect ratio = w/h
  - **Hu Moments**: Bất biến với rotation, scale (dùng để mô tả hình dạng)

- **Code nâng cấp (3-4h)**:
  ```python
  def extract_features(contour):
      area = cv2.contourArea(contour)
      perimeter = cv2.arcLength(contour, True)
      circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
      x,y,w,h = cv2.boundingRect(contour)
      aspect_ratio = w / h if h > 0 else 0
      hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
      return [area, perimeter, circularity, aspect_ratio] + list(hu_moments)
  ```

- **Output**: CSV file chứa features của mỗi contour, filter contour theo rule-based (ví dụ: circularity < 0.5 → có thể là khuyết tật dạng crack)

**📌 Keywords tuần 1-2**: `CLAHE`, `Adaptive Threshold`, `Morphology`, `Contour Features`, `Hu Moments`

---

### **TUẦN 3-4: Deep Learning Basics (CNN cơ bản cho Classification)**

**Mục tiêu**: Học CNN để phân loại ảnh (OK vs NG) hoặc phân loại từng contour

#### Tuần 3: CNN Architecture & Transfer Learning
- **Đọc từ PDF**:
  - **Chương 3: Linear Regression** (trang 49-59) → hiểu Loss function (MSE, MAE), Gradient Descent, Regularization
  - **Chương 4-6: Neural Network basics, Backpropagation, CNN** (nếu có) → hiểu Convolution, Pooling, Activation (ReLU)
  
- **Đọc thêm**:
  - `CNN explained simple` (3Blue1Brown YouTube hoặc blog)
  - `Transfer Learning PyTorch/TensorFlow`
  - `ResNet MobileNet architecture`

- **Key concepts**:
  - **Convolution**: Kernel/filter trích xuất feature từ ảnh
    - Formula: `output_size = (input - kernel + 2×padding) / stride + 1`
    - Receptive field: Vùng ảnh mà mỗi neuron "nhìn thấy"
  - **Pooling**: MaxPooling/AvgPooling giảm kích thước spatial
  - **Transfer Learning**: Dùng pretrained model (ResNet, MobileNet) → fine-tune trên dataset nhỏ của bạn
  - **Loss**: CrossEntropyLoss (classification), Binary CrossEntropy (binary classification)

- **Code (4-5h)**:
  ```python
  # classifier_v1.py - Binary classification (OK vs Defect)
  import torch
  import torchvision.models as models
  
  model = models.resnet18(pretrained=True)
  model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: OK, NG
  
  # Training loop (simplified)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  ```

- **Output**: Model phân loại ảnh toàn bộ (full image) thành OK/NG với accuracy ~80-90%

#### Tuần 4: Data Augmentation & Training
- **Đọc từ PDF**:
  - **Chương Regularization** (trang 49-59): Dropout, L2 weight decay, BatchNorm, Early Stopping
  - **Chương Training & Optimization** (trang 27-35): Learning rate scheduling, Adam vs SGD

- **Đọc thêm**:
  - `Data Augmentation for small dataset`
  - `imgaug albumentations library`
  - `Learning rate scheduler PyTorch`

- **Key concepts**:
  - **Augmentation**: Rotation (±10°), Horizontal flip, Brightness/Contrast, Noise (cẩn thận với vertical flip cho ảnh X-ray)
  - **Regularization**: 
    - Dropout (0.3-0.5) ở fully-connected layers
    - L2 weight decay (1e-4)
    - BatchNorm (sau Conv, trước ReLU)
  - **LR Scheduler**: ReduceLROnPlateau (giảm LR khi val_loss không cải thiện), CosineAnnealing

- **Code (4-5h)**:
  ```python
  import albumentations as A
  
  transform = A.Compose([
      A.Rotate(limit=15, p=0.5),
      A.HorizontalFlip(p=0.5),
      A.RandomBrightnessContrast(p=0.3),
      A.GaussNoise(p=0.2)
  ])
  
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                           factor=0.5, patience=5)
  ```

- **Output**: Model với augmentation + regularization, accuracy cải thiện ~5-10%, training curve (loss/accuracy plot)

**📌 Keywords tuần 3-4**: `CNN`, `Convolution`, `Pooling`, `Transfer Learning`, `ResNet`, `MobileNet`, `Data Augmentation`, `Dropout`, `Learning Rate Scheduler`

---

### **TUẦN 5-6: Object Detection (Phát hiện & Định vị khuyết tật)**

**Mục tiêu**: Dùng YOLO hoặc Faster R-CNN để phát hiện vị trí khuyết tật (bounding box)

#### Tuần 5: YOLO Basics & Labeling
- **Đọc từ PDF**: 
  - Nếu có chương Detection → đọc IoU, mAP, Anchor boxes
  - Nếu không có → Google search

- **Đọc thêm**:
  - `YOLO object detection explained`
  - `YOLOv8 Ultralytics tutorial`
  - `LabelImg annotation tool`
  - `COCO dataset format`

- **Key concepts**:
  - **Object Detection**: Classify + Localize (bounding box)
  - **IoU (Intersection over Union)**: Metric đo overlap giữa predicted box và ground truth
    - Formula: `IoU = Area(overlap) / Area(union)`
    - IoU > 0.5 → good detection
  - **mAP (mean Average Precision)**: Metric tổng hợp cho detection (mAP@0.5, mAP@0.5:0.95)
  - **YOLO**: Single-stage detector (nhanh), chia ảnh thành grid, mỗi cell dự đoán bounding box + class
  - **Anchor boxes**: Predefined bounding box shapes (học từ dataset)

- **Code (5-6h)**:
  - Label ~50-100 ảnh bằng LabelImg (format YOLO txt)
  - Train YOLOv8:
  ```python
  from ultralytics import YOLO
  
  model = YOLO('yolov8n.pt')  # nano model (nhẹ, nhanh)
  model.train(data='data.yaml', epochs=50, imgsz=640, batch=8)
  ```

- **Output**: Model detect bounding box của khuyết tật, mAP@0.5 ~60-70% (tùy data quality)

#### Tuần 6: Model Comparison & Optimization
- **Đọc thêm**:
  - `YOLO vs Faster R-CNN comparison`
  - `Model quantization INT8 FP16`
  - `ONNX export inference speed`

- **Key concepts**:
  - **YOLO**: Nhanh (real-time), accuracy trung bình
  - **Faster R-CNN**: Chậm hơn, accuracy cao hơn (two-stage)
  - **Model size**: YOLOv8n (nano) ~3MB, YOLOv8s (small) ~11MB, YOLOv8m (medium) ~26MB
  - **Inference speed**: FPS (frames per second) trên CPU/GPU

- **Code (4-5h)**:
  - So sánh YOLOv8n vs YOLOv8s
  - Export to ONNX:
  ```python
  model.export(format='onnx')  # for deployment
  ```
  - Test inference speed

- **Output**: Báo cáo so sánh (mAP, FPS, model size), chọn model phù hợp

**📌 Keywords tuần 5-6**: `YOLO`, `Object Detection`, `IoU`, `mAP`, `Bounding Box`, `Anchor`, `LabelImg`, `ONNX`

---

### **TUẦN 7: Semantic Segmentation (Nâng cao - phân đoạn pixel-level)**

**Mục tiêu**: Dùng U-Net để phân đoạn khuyết tật (chính xác hơn bounding box)

- **Đọc thêm**:
  - `U-Net architecture explained`
  - `Semantic Segmentation vs Instance Segmentation`
  - `Dice Loss IoU metric segmentation`

- **Key concepts**:
  - **Semantic Segmentation**: Classify từng pixel (background vs defect)
  - **U-Net**: Encoder-Decoder architecture với skip connections (tốt cho medical/industrial images)
  - **Dice Loss**: Loss function cho segmentation (xử lý class imbalance tốt)
    - Formula: `Dice = 2×|A∩B| / (|A|+|B|)`
  - **IoU/Dice score**: Metric đánh giá segmentation

- **Code (5-6h)**:
  ```python
  # u_net.py (simplified)
  import segmentation_models_pytorch as smp
  
  model = smp.Unet(
      encoder_name="resnet34",
      encoder_weights="imagenet",
      in_channels=1,  # grayscale
      classes=1,       # binary segmentation
  )
  
  loss = smp.losses.DiceLoss(mode='binary')
  ```

- **Output**: Mask phân đoạn chính xác vùng khuyết tật (pixel-level), Dice score ~0.75-0.85

**📌 Keywords tuần 7**: `U-Net`, `Semantic Segmentation`, `Dice Loss`, `Pixel-wise Classification`, `Encoder-Decoder`

---

### **TUẦN 8: Deployment & System Integration**

**Mục tiêu**: Đóng gói model thành API/app, tối ưu inference speed

- **Đọc từ PDF**:
  - **Chương Deployment** (nếu có): SavedModel, ONNX, TensorRT

- **Đọc thêm**:
  - `FastAPI machine learning deployment`
  - `OpenVINO Intel optimization`
  - `Docker containerization ML model`

- **Key concepts**:
  - **Model export**: `.pt` (PyTorch), `.onnx` (cross-framework), `.engine` (TensorRT)
  - **Inference optimization**: 
    - Quantization (FP32 → FP16/INT8) → 2-4× faster
    - Batch inference (process multiple images at once)
    - OpenVINO (Intel) / TensorRT (NVIDIA) for hardware acceleration
  - **API**: FastAPI/Flask serve model qua HTTP
  - **Docker**: Container hóa app (model + dependencies)

- **Code (6-8h)**:
  ```python
  # api.py
  from fastapi import FastAPI, UploadFile
  import cv2
  import numpy as np
  from ultralytics import YOLO
  
  app = FastAPI()
  model = YOLO('best.pt')
  
  @app.post("/predict")
  async def predict(file: UploadFile):
      img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
      results = model(img)
      return {"boxes": results[0].boxes.xyxy.tolist()}
  ```

- **Output**: 
  - API endpoint nhận ảnh, trả về predictions
  - Docker image chạy được trên máy khác
  - Báo cáo tốc độ inference (ms/image)

**📌 Keywords tuần 8**: `FastAPI`, `ONNX`, `TensorRT`, `OpenVINO`, `Quantization`, `Docker`, `Model Deployment`

---

## 📊 Tổng kết các mô hình cần tìm hiểu (theo thứ tự dễ → khó)

| Tuần | Mô hình/Kỹ thuật | Mục đích | Độ khó |
|------|------------------|----------|--------|
| 1-2 | Classical CV (CLAHE, Morphology) | Preprocessing | ⭐ |
| 3-4 | **ResNet/MobileNet** (Transfer Learning) | Image Classification | ⭐⭐ |
| 5-6 | **YOLOv8** (Object Detection) | Detect bounding box | ⭐⭐⭐ |
| 6 (optional) | **Faster R-CNN** | Detection accuracy cao hơn | ⭐⭐⭐⭐ |
| 7 | **U-Net** (Semantic Segmentation) | Phân đoạn pixel-level | ⭐⭐⭐⭐ |
| 7 (optional) | **Mask R-CNN** | Instance Segmentation | ⭐⭐⭐⭐⭐ |

---

## 🔑 Key Concepts cần master (Google search keywords)

### Week 1-2 (Classical CV)
- `CLAHE contrast enhancement`
- `Otsu vs Adaptive Threshold`
- `Morphological operations OpenCV`
- `Contour features aspect ratio circularity`

### Week 3-4 (CNN Basics)
- `Convolution explained`
- `Receptive field CNN`
- `Transfer Learning fine-tuning`
- `Data Augmentation techniques`
- `Dropout BatchNorm Regularization`
- `Learning rate scheduler PyTorch`

### Week 5-6 (Object Detection)
- `YOLO architecture how it works`
- `IoU calculation object detection`
- `mAP metric explained`
- `Anchor boxes YOLO`
- `Non-Maximum Suppression NMS`

### Week 7 (Segmentation)
- `U-Net architecture skip connections`
- `Dice Loss vs BCE Loss`
- `Semantic vs Instance Segmentation`

### Week 8 (Deployment)
- `ONNX model export`
- `Model quantization FP16 INT8`
- `FastAPI machine learning tutorial`
- `Docker containerize ML model`

---

## 💡 Tips để không bị ngộp

1. **Mỗi tuần chỉ focus 1 topic chính** (ví dụ: tuần 3 = CNN basics, đừng nhảy sang YOLO)
2. **Code ngay sau khi đọc lý thuyết** (30 phút đọc → 30 phút code)
3. **Lưu code + notes vào Git** (commit mỗi tuần để theo dõi progress)
4. **Đọc PDF chương tương ứng trước, sau đó Google search chi tiết**
5. **Ưu tiên practical (code) hơn theory sâu** (ví dụ: hiểu cách dùng YOLO > hiểu toán đằng sau YOLO)

---

## 📝 Daily Schedule Template

**Thứ 2-6 (1h/ngày)**:
- 20 phút: Đọc lý thuyết (PDF + blog)
- 30 phút: Code/experiment
- 10 phút: Note lại key points + commit code

**Thứ 7-CN (3h/ngày)**:
- 1h: Đọc lý thuyết sâu hơn (paper, tutorial)
- 1.5h: Code project chính (train model, test)
- 30 phút: Review tuần + chuẩn bị tuần sau

---

## 🎯 Deliverables cuối 8 tuần

1. ✅ **Preprocessing pipeline** nâng cấp (CLAHE + Adaptive Threshold + Morphology)
2. ✅ **Classifier** (ResNet/MobileNet) phân loại OK/NG với accuracy >85%
3. ✅ **Detector** (YOLOv8) phát hiện bounding box với mAP@0.5 >70%
4. ✅ **(Optional)** **Segmentation model** (U-Net) với Dice >0.75
5. ✅ **API deployment** (FastAPI) + Docker container
6. ✅ **Báo cáo so sánh** các mô hình (accuracy, speed, size)

---

**Next step**: Bắt đầu tuần 1 → upgrade `test02.py` với CLAHE + Adaptive Threshold. Bạn muốn tôi tạo file `test02_v2.py` mẫu không?
