# Lộ trình học Computer Vision (từ dễ → khó)

**Mục tiêu**: Nâng cấp từ code phát hiện contour cơ bản (`test02.py`) lên hệ thống phát hiện khuyết tật tự động (detection/classification) với Deep Learning.

**Nguyên tắc**: Kiến thức từ dễ → khó; học từng giai đoạn, vừa đọc lý thuyết vừa code thực hành.

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

## 🗓️ Lộ trình học (Easy → Hard)

### **Giai đoạn 1: Classical Computer Vision (Nền tảng xử lý ảnh)**

**Mục tiêu**: Nâng cấp preprocessing pipeline (tiền xử lý ảnh tốt hơn)

#### Phần 1: Adaptive Thresholding & CLAHE
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

#### Phần 2: Feature Engineering (Geometric features)
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

**📌 Keywords giai đoạn 1**: `CLAHE`, `Adaptive Threshold`, `Morphology`, `Contour Features`, `Hu Moments`

---

### **Giai đoạn 2: Deep Learning Basics (CNN cơ bản cho Classification)**

**Mục tiêu**: Học CNN để phân loại ảnh (OK vs NG) hoặc phân loại từng contour

#### Phần 1: CNN Architecture & Transfer Learning
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

#### Phần 2: Data Augmentation & Training
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

**📌 Keywords giai đoạn 2**: `CNN`, `Convolution`, `Pooling`, `Transfer Learning`, `ResNet`, `MobileNet`, `Data Augmentation`, `Dropout`, `Learning Rate Scheduler`

---

### **Giai đoạn 3: Object Detection (Phát hiện & Định vị khuyết tật)**

**Mục tiêu**: Dùng YOLO hoặc Faster R-CNN để phát hiện vị trí khuyết tật (bounding box)

#### Phần 1: YOLO Basics & Labeling
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
  - Train YOLOv8/v11/RT-DETR (syntax giống hệt nhau):
  ```python
  from ultralytics import YOLO
  
  # YOLOv8 - Baseline
  model = YOLO('yolov8n.pt')  # nano model (nhẹ, nhanh)
  
  # YOLOv11 - More accurate (recommended nếu muốn điểm cao)
  model = YOLO('yolo11n.pt')  # hoặc yolo11s.pt
  
  # RT-DETR - Fastest inference (transformer-based)
  model = YOLO('rtdetr-l.pt')  # hoặc rtdetr-x.pt
  
  # Training (cùng syntax cho cả 3 models)
  model.train(data='data.yaml', epochs=50, imgsz=640, batch=8)
  ```

- **Output**: Model detect bounding box của khuyết tật, mAP@0.5 ~60-70% (tùy data quality)

#### Phần 2: Model Comparison & Optimization
- **Đọc thêm**:
  - `YOLO vs Faster R-CNN comparison`
  - `Model quantization INT8 FP16`
  - `ONNX export inference speed`

- **Key concepts**:
  - **YOLOv8**: Nhanh (real-time), accuracy trung bình, model nhỏ (3-11MB)
  - **YOLOv11**: Giống YOLOv8 nhưng mAP cao hơn ~5%, model size tăng nhẹ (5-12MB)
  - **RT-DETR**: Transformer-based, fastest inference (~2x faster GPU), nhưng model lớn hơn (20MB)
  - **Inference speed**: FPS (frames per second) trên CPU/GPU
  - **Trade-offs**: YOLOv8 (balance), YOLOv11 (accuracy), RT-DETR (speed)

- **Code (4-5h)**:
  - So sánh YOLOv8n vs YOLOv8s (baseline)
  - Nếu dư thời gian: thêm YOLOv11n, RT-DETR-L
  - Export to ONNX:
  ```python
  model.export(format='onnx')  # for deployment
  
  # Benchmark inference speed
  import time
  img = cv2.imread('test.jpg')
  start = time.time()
  results = model(img)
  fps = 1 / (time.time() - start)
  print(f"FPS: {fps:.2f}")
  ```
  - Test inference speed (CPU/GPU)

- **Output**: Báo cáo so sánh (mAP, FPS, model size), chọn model phù hợp

**📌 Keywords giai đoạn 3**: `YOLO`, `YOLOv11`, `RT-DETR`, `Object Detection`, `IoU`, `mAP`, `Bounding Box`, `Anchor-free`, `Transformer Detection`, `LabelImg`, `ONNX`

---

### **Giai đoạn 4: Semantic Segmentation (Nâng cao - phân đoạn pixel-level)**

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
  
  # U-Net - Standard choice (best accuracy)
  model = smp.Unet(
      encoder_name="resnet34",
      encoder_weights="imagenet",
      in_channels=1,  # grayscale
      classes=1,       # binary segmentation
  )
  
  # DeepLabV3+ - Faster alternative (nếu dư thời gian)
  model = smp.DeepLabV3Plus(
      encoder_name="resnet50",       # hoặc mobilenet_v2 (fastest)
      encoder_weights="imagenet",
      in_channels=1,
      classes=1
  )
  
  loss = smp.losses.DiceLoss(mode='binary')
  
  # Comparison: U-Net vs DeepLabV3+
  # - U-Net: Dice ~0.78-0.85, inference ~200-400ms (CPU)
  # - DeepLabV3+: Dice ~0.75-0.82, inference ~150-300ms (CPU)
  # Trade-off: DeepLabV3+ 30-40% faster, 2-3% Dice drop
  ```

- **Output**: Mask phân đoạn chính xác vùng khuyết tật (pixel-level), Dice score ~0.75-0.85

**📌 Keywords giai đoạn 4**: `U-Net`, `DeepLabV3+`, `Semantic Segmentation`, `Dice Loss`, `Pixel-wise Classification`, `Encoder-Decoder`, `ASPP`

---



---

## 📊 Tổng kết các mô hình cần tìm hiểu (theo thứ tự dễ → khó)

| Giai đoạn | Mô hình/Kỹ thuật | Mục đích | Độ khó |
|------|------------------|----------|--------|
| 1 | Classical CV (CLAHE, Morphology) | Preprocessing | ⭐ |
| 2 | **ResNet/MobileNet** (Transfer Learning) | Image Classification | ⭐⭐ |
| 3 | **YOLOv8** (Object Detection) | Detect bounding box | ⭐⭐⭐ |
| 3 (optional) | **YOLOv11 / RT-DETR** | More accurate / Faster detection | ⭐⭐⭐ |
| 4 | **U-Net** (Semantic Segmentation) | Phân đoạn pixel-level | ⭐⭐⭐⭐ |
| 4 (optional) | **DeepLabV3+** | Faster segmentation | ⭐⭐⭐⭐ |

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

1. **Mỗi giai đoạn chỉ focus 1 topic chính** (ví dụ: CNN basics, đừng nhảy sang YOLO ngay)
2. **Code ngay sau khi đọc lý thuyết** — học từng phần nhỏ
3. **Lưu code + notes vào Git** — commit thường xuyên để theo dõi progress
4. **Đọc PDF chương tương ứng trước, sau đó Google search chi tiết**
5. **Ưu tiên practical (code) hơn theory sâu** (ví dụ: hiểu cách dùng YOLO > hiểu toán đằng sau YOLO)
6. **Không cần làm theo thứ tự cứng nhắc** — nhảy giai đoạn nếu cần thiết cho project

---

## 🎯 Deliverables cuối cùng

1. ✅ **Preprocessing pipeline** nâng cấp (CLAHE + Adaptive Threshold + Morphology)
2. ✅ **Classifier** (ResNet/MobileNet) phân loại OK/NG với accuracy >85%
3. ✅ **Detector** (YOLOv8) phát hiện bounding box với mAP@0.5 >70%
4. ✅ **(Optional)** **Segmentation model** (U-Net) với Dice >0.75
5. ✅ **API deployment** (FastAPI) + Docker container
6. ✅ **Báo cáo so sánh** các mô hình (accuracy, speed, size)

---

## 🏥 Ứng dụng cuối kỳ: X-ray Defect Detection System

**Yêu cầu tổng hợp** (tích hợp tất cả kiến thức từ 8 tuần):

### Tính năng chính
- ✅ **Nhận ảnh X-ray đầu vào** (upload qua web UI hoặc API)
- ✅ **Phát hiện vùng khuyết tật** bằng **YOLOv8** (bounding box)
- ✅ **Phân đoạn vùng khuyết tật** bằng **U-Net** (pixel-level mask)
- ✅ **Tính toán tỷ lệ % khuyết tật**:
  ```python
  defect_ratio = (số pixel khuyết tật / tổng số pixel ROI) × 100%
  ```
- ✅ **Hiển thị kết quả** qua giao diện đơn giản:
  - Input image (original)
  - YOLOv8 detection (bounding boxes + confidence scores)
  - U-Net segmentation (overlay mask màu đỏ/vàng)
  - Metrics: % khuyết tật, số lượng defects, inference time

### Kiến trúc hệ thống
```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Frontend   │ ───► │   Backend    │ ───► │   Models    │
│  (Streamlit │      │  (FastAPI)   │      │ YOLOv8+UNet │
│   hoặc      │ ◄─── │              │ ◄─── │             │
│   Gradio)   │      │  Inference   │      │   ONNX      │
└─────────────┘      └──────────────┘      └─────────────┘
```

### Implementation Plan

#### 1. Model Training & Export
```python
# 1. Train YOLOv8 (detection)
from ultralytics import YOLO
model_yolo = YOLO('yolov8n.pt')
model_yolo.train(data='xray_defect.yaml', epochs=50, imgsz=640)
model_yolo.export(format='onnx')  # → best_yolo.onnx

# 2. Train U-Net (segmentation)
import segmentation_models_pytorch as smp
model_unet = smp.Unet(encoder_name="resnet34", classes=1)
# ... training loop ...
torch.onnx.export(model_unet, dummy_input, 'unet.onnx')  # → unet.onnx
```

#### 2. Inference Pipeline
```python
# inference.py
import cv2
import numpy as np
import onnxruntime as ort

class XrayDefectDetector:
    def __init__(self, yolo_path, unet_path):
        self.yolo_session = ort.InferenceSession(yolo_path)
        self.unet_session = ort.InferenceSession(unet_path)
    
    def detect_and_segment(self, image):
        # Step 1: YOLOv8 detection
        boxes, scores = self.run_yolo(image)
        
        # Step 2: U-Net segmentation (crop ROI từ YOLO boxes)
        masks = []
        for box in boxes:
            x1, y1, x2, y2 = box
            roi = image[y1:y2, x1:x2]
            mask = self.run_unet(roi)
            masks.append(mask)
        
        # Step 3: Calculate defect ratio
        total_defect_pixels = sum([mask.sum() for mask in masks])
        total_roi_pixels = sum([(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes])
        defect_ratio = (total_defect_pixels / total_roi_pixels) * 100 if total_roi_pixels > 0 else 0
        
        return {
            'boxes': boxes,
            'scores': scores,
            'masks': masks,
            'defect_ratio': defect_ratio,
            'num_defects': len(boxes)
        }
```

#### 3. API Backend (FastAPI)
```python
# api.py
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from inference import XrayDefectDetector

app = FastAPI()
detector = XrayDefectDetector('best_yolo.onnx', 'unet.onnx')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Run inference
    results = detector.detect_and_segment(img)
    
    # Visualize
    vis_img = visualize_results(img, results)
    
    return {
        "num_defects": results['num_defects'],
        "defect_ratio": f"{results['defect_ratio']:.2f}%",
        "boxes": results['boxes'].tolist(),
        "scores": results['scores'].tolist(),
        "visualization": encode_image_base64(vis_img)
    }
```

#### 4. Frontend UI (Streamlit hoặc Gradio)
```python
# app_streamlit.py
import streamlit as st
import requests
from PIL import Image

st.title("X-ray Defect Detection System")

uploaded_file = st.file_uploader("Upload X-ray image", type=['png', 'jpg'])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption='Original X-ray', use_column_width=True)
    
    # Send to API
    files = {'file': uploaded_file.getvalue()}
    response = requests.post('http://localhost:8000/predict', files=files)
    results = response.json()
    
    # Display results
    st.subheader("Detection Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Defects Found", results['num_defects'])
    col2.metric("Defect Ratio", results['defect_ratio'])
    col3.metric("Confidence", f"{max(results['scores'])*100:.1f}%")
    
    # Display visualization
    st.image(results['visualization'], caption='YOLOv8 + U-Net Results', use_column_width=True)
```

### Báo cáo so sánh YOLOv8 vs U-Net

#### Tiêu chí đánh giá

| Tiêu chí | YOLOv8 | YOLOv11 | RT-DETR | U-Net | DeepLabV3+ | Ghi chú |
|----------|---------|---------|---------|-------|------------|---------|
| **Mục đích** | Bounding box | Bounding box | Bounding box | Pixel-level | Pixel-level | - |
| **Độ chính xác** | mAP@0.5: 70-80% | mAP@0.5: 75-85% | mAP@0.5: 72-82% | Dice: 0.75-0.85 | Dice: 0.73-0.83 | YOLOv11 accurate nhất detection |
| **Tốc độ (CPU)** | ~50-100ms | ~50-100ms | ~30-60ms | ~200-400ms | ~150-300ms | RT-DETR nhanh nhất |
| **Tốc độ (GPU)** | ~10-20ms | ~10-20ms | ~5-10ms | ~30-50ms | ~20-35ms | RT-DETR fastest inference |
| **Model size** | 3MB (nano) | 5MB (nano) | 20MB (L) | 20-50MB | 25-60MB | YOLO nhỏ nhất |
| **Ease of use** | ✅ Rất dễ | ✅ Rất dễ | ✅ Rất dễ | ⭐⭐ Khá | ⭐⭐ Khá | Detection models dễ hơn |
| **Use case** | Screening nhanh | Accuracy cao | Real-time | Tính % defect | Fast segmentation | Tùy yêu cầu |

#### Ưu điểm

**Detection Models (YOLOv8/v11/RT-DETR)**:
- ✅ Rất nhanh (real-time trên GPU)
- ✅ Model nhỏ gọn (YOLOv8: 3MB, YOLOv11: 5MB)
- ✅ Dễ train (ít data, cùng syntax)
- ✅ Tốt cho counting (đếm số lượng defects)
- ✅ **YOLOv11**: Accurate nhất (mAP cao hơn YOLOv8 ~5%)
- ✅ **RT-DETR**: Nhanh nhất inference (~2x faster than YOLO)

**Segmentation Models (U-Net/DeepLabV3+)**:
- ✅ Chính xác pixel-level (tính % defect chính xác)
- ✅ Phân đoạn biên rõ ràng
- ✅ Tốt cho medical/industrial images
- ✅ Có thể phân biệt defects chồng lấn
- ✅ **DeepLabV3+**: Nhanh hơn U-Net ~30-40%, pretrained encoders tốt

#### Nhược điểm

**Detection Models (YOLOv8/v11/RT-DETR)**:
- ❌ Chỉ bounding box (không chính xác về diện tích)
- ❌ Khó phân đoạn defects có hình dạng phức tạp
- ❌ Box overlap khi defects gần nhau
- ❌ **RT-DETR**: Model size lớn hơn YOLO (~20MB vs 3-5MB)

**Segmentation Models (U-Net/DeepLabV3+)**:
- ❌ Chậm hơn detection (2-4x)
- ❌ Model lớn hơn (~25-60MB)
- ❌ Cần nhiều data labeled (pixel-wise masks)
- ❌ Khó train với small dataset
- ❌ **DeepLabV3+**: Dice score thấp hơn U-Net ~2-3% (trade-off speed vs accuracy)

#### Kết luận & Khuyến nghị

**Chiến lược kết hợp (best of both worlds)**:
1. **Stage 1 (Detection)**: YOLOv8/v11/RT-DETR → detect ROI nhanh
2. **Stage 2 (Segmentation)**: U-Net/DeepLabV3+ → segment chi tiết trong ROI
3. **Lợi ích**: Tốc độ detection + độ chính xác segmentation

**Chọn model phù hợp**:
- **YOLOv8**: Baseline tốt, nhỏ gọn (recommended bắt đầu)
- **YOLOv11**: Accuracy cao nhất detection (+5% mAP) - nếu muốn điểm cao
- **RT-DETR**: Fastest inference (real-time edge devices)
- **U-Net**: Accuracy cao nhất segmentation (standard choice)
- **DeepLabV3+**: Faster segmentation, trade-off 2-3% Dice cho 30-40% speed

**Đề xuất cho báo cáo (tăng điểm)**:
- **Minimum (pass)**: YOLOv8 + U-Net
- **Better (điểm cao)**: Thêm YOLOv11 hoặc RT-DETR + so sánh detection models
- **Best (điểm rất cao)**: Train cả 4-5 models → comparative analysis table với mAP/Dice/FPS/Size

---

---

## 🔧 Giai đoạn 5 (Cuối cùng): UI & Deployment

**Làm sau khi đã train xong model detection/segmentation**

### Model Export & Optimization

- **Đọc thêm**:
  - `ONNX model export PyTorch`
  - `Model quantization FP16 INT8`
  - `TensorRT OpenVINO optimization`

- **Key concepts**:
  - **Model export**: `.pt` (PyTorch) → `.onnx` (cross-framework) → `.engine` (TensorRT)
  - **Inference optimization**: 
    - Quantization (FP32 → FP16/INT8) → 2-4× faster
    - Batch inference (process multiple images at once)
    - OpenVINO (Intel) / TensorRT (NVIDIA) for hardware acceleration

- **Code**:
  ```python
  # Export YOLO to ONNX
  from ultralytics import YOLO
  model = YOLO('best.pt')
  model.export(format='onnx')  # → best.onnx
  
  # Export U-Net to ONNX
  import torch
  torch.onnx.export(unet_model, dummy_input, 'unet.onnx')
  ```

**📌 Keywords**: `ONNX`, `TensorRT`, `OpenVINO`, `Quantization`, `Model Export`

---

### API Backend (FastAPI)

- **Đọc thêm**:
  - `FastAPI machine learning deployment`
  - `Docker containerization ML model`

- **Code**:
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

**📌 Keywords**: `FastAPI`, `Docker`, `API Deployment`

---

### Frontend UI (Streamlit/Gradio)

- **Code mẫu** (Streamlit):
  ```python
  # app_streamlit.py
  import streamlit as st
  import requests
  from PIL import Image
  
  st.title("X-ray Defect Detection")
  uploaded = st.file_uploader("Upload X-ray", type=['png', 'jpg'])
  
  if uploaded:
      image = Image.open(uploaded)
      st.image(image, caption='Original', use_column_width=True)
      
      # Send to API
      files = {'file': uploaded.getvalue()}
      response = requests.post('http://localhost:8000/predict', files=files)
      results = response.json()
      
      st.metric("Defects Found", results['num_defects'])
      st.metric("Defect Ratio", results['defect_ratio'])
  ```

**📌 Keywords**: `Streamlit`, `Gradio`, `Web UI`

---

## 📋 Checklist hoàn thành project

### Core (Ưu tiên - đủ để pass)
- [ ] **Giai đoạn 1-2**: Preprocessing + Classification (baseline)
- [ ] **Giai đoạn 3**: YOLOv8 training + evaluation (mAP >70%)
- [ ] **Giai đoạn 4**: U-Net training + evaluation (Dice >0.75)
- [ ] **Inference pipeline**: YOLO → U-Net (script đã có: `scripts/inference_pipeline.py`)
- [ ] **Báo cáo so sánh**: YOLOv8 vs U-Net (accuracy, speed, model size)

### Advanced (Nếu dư thời gian - để điểm cao hơn)
- [ ] **Detection comparison**: Train thêm YOLOv11 hoặc RT-DETR → compare với YOLOv8
- [ ] **Segmentation comparison**: Train thêm DeepLabV3+ → compare với U-Net
- [ ] **Comparative analysis**: Table so sánh mAP/Dice/FPS/Model Size của 4-5 models
- [ ] **Ablation study**: Test các encoder backbones khác nhau (ResNet34 vs MobileNetV2 vs EfficientNet)

### Polish (Làm sau)
- [ ] **Model export**: Convert to ONNX/TensorRT
- [ ] **API backend**: FastAPI serve model
- [ ] **Frontend UI**: Streamlit/Gradio
- [ ] **Docker deployment**: Container hóa app
- [ ] **Demo video**: Upload ảnh → hiển thị kết quả

---

**Next step**: Bắt đầu giai đoạn 1 → upgrade `test02.py` với CLAHE + Adaptive Threshold. Hoặc nhảy thẳng sang giai đoạn 3-4 nếu muốn train YOLO/U-Net trước (recommended: focus detection/segmentation trước, UI sau).
