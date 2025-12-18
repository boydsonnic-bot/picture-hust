"""
HYBRID TRAINING - MOBILENETV2 PARTIAL FREEZE
Phương pháp: Đóng băng backbone, unfreeze vài block cuối + train classifier
Mục đích: Cân bằng giữa tốc độ training và accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
import time
import random
from contextlib import nullcontext  # Dùng để handle context manager khi không dùng AMP
from tqdm import tqdm  # Progress bar đẹp
from torch.amp import autocast, GradScaler  # Mixed Precision Training
from data import Cv2PreprocessDataset, transform_config

# ============================================================
# PHẦN 1: CẤU HÌNH GLOBAL
# Lý do đặt ngoài: Windows multiprocessing cần import lại file
# → Biến phải ở global scope để workers thấy được
# ============================================================

# --- 1.1. Hardware Config ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Giải thích: 
# - torch.cuda.is_available(): Kiểm tra có GPU NVIDIA + CUDA không
# - Nếu CÓ: dùng 'cuda' (nhanh gấp 10-50 lần)
# - Nếu KHÔNG: dùng 'cpu' (chậm nhưng vẫn chạy được)

# --- 1.2. Data Loading Config ---
BATCH_SIZE = 32
# Giải thích:
# - Mỗi lần đưa 32 ảnh vào GPU để train
# - Tại sao 32? Vừa đủ cho GPU 4GB, vừa đủ để gradient ổn định
# - Nhỏ hơn (8-16): Chậm, nhưng ít VRAM
# - Lớn hơn (64-128): Nhanh, nhưng cần GPU mạnh

NUM_WORKERS = 4
# Giải thích:
# - Số thread CPU load ảnh song song (không block GPU)
# - 0: Main thread load (chậm, GPU phải chờ)
# - 4: 4 threads load song song → GPU chạy liên tục
# - Rule: cpu_count // 2, max 8
# - Lưu ý: Windows cần if __name__ guard để không bị lỗi

NUM_EPOCHS = 50
# Giải thích:
# - Số lần model "nhìn" toàn bộ dataset
# - 1 epoch = 1 lần quét hết tất cả ảnh training
# - 50 epochs: Đủ để model học tốt, có early stopping nên có thể dừng sớm

# --- 1.3. Learning Rate Config ---
HEAD_LR = 1e-3  # = 0.001
# Giải thích:
# - Learning rate cho classifier (head) - phần MỚI cần học nhiều
# - 1e-3 = 0.001: Tốc độ học vừa phải
# - Quá cao (0.1): Model "nhảy lung tung", không hội tụ
# - Quá thấp (0.00001): Học quá chậm, tốn thời gian

BACKBONE_LR = 1e-4  # = 0.0001
# Giải thích:
# - Learning rate cho backbone (features) - phần ĐÃ TRAIN SẴN
# - Thấp hơn HEAD_LR gấp 10 lần vì backbone đã học tốt rồi
# - Chỉ cần "tinh chỉnh" nhẹ, không muốn phá hỏng kiến thức cũ

WEIGHT_DECAY = 1e-4  # = 0.0001
# Giải thích:
# - L2 regularization - "phạt" weights quá lớn
# - Công thức: loss_total = loss + weight_decay * sum(w²)
# - Mục đích: Tránh overfitting, model tổng quát hơn
# - 1e-4: Giá trị chuẩn cho transfer learning

# --- 1.4. Model Config ---
UNFREEZE_LAST_N_BLOCKS = 2
# Giải thích:
# - MobileNetV2 có 17 blocks (features[0] đến features[16])
# - Mở khóa 2 blocks CUỐI (features[15], features[16]) để học
# - Tại sao? Blocks cuối học "high-level features" gần với task
# - Trade-off:
#   * Unfreeze 0: Nhanh nhất, accuracy ~85%
#   * Unfreeze 2: Chậm hơn chút, accuracy ~88% ← DÙNG
#   * Unfreeze 17: Chậm nhất, accuracy ~90%, dễ overfit

NUM_CLASSES = 4
# Giải thích:
# - Số lớp cần phân loại: CR, LP, OK, PO
# - ImageNet gốc: 1000 classes
# - Chúng ta: 4 classes → Thay classifier layer

DATA_PATH = r'C:\project\picture-hust\data\train'
# Giải thích:
# - r'...' : Raw string, tránh lỗi với backslash \ trên Windows
# - Đường dẫn đến folder chứa ảnh training

# ============================================================
# PHẦN 2: MAIN GUARD - BẮT BUỘC CHO MULTIPROCESSING
# Lý do: Tránh "recursive spawn" trên Windows
# ============================================================
if __name__ == '__main__':
    # Giải thích if __name__ == '__main__':
    # - Khi chạy: python train.py → __name__ = '__main__' → Vào đây
    # - Khi import: import train → __name__ = 'train' → KHÔNG vào
    # - Windows spawn workers → import lại file → Không tạo DataLoader lại
    # → Tránh đệ quy vô hạn!
    
    # ============================================================
    # PHẦN 2.1: REPRODUCIBILITY - ĐẢM BẢO KẾT QUẢ LẶP LẠI ĐƯỢC
    # ============================================================
    SEED = 42
    # Giải thích:
    # - Seed = "hạt giống" cho random number generator
    # - Dùng cùng seed → cùng kết quả random → kết quả lặp lại được
    # - 42: Số phổ biến (từ "The Hitchhiker's Guide to the Galaxy")
    
    random.seed(SEED)
    # Giải thích:
    # - Set seed cho module random của Python (shuffle, random.choice...)
    
    torch.manual_seed(SEED)
    # Giải thích:
    # - Set seed cho PyTorch CPU operations
    # - Ảnh hưởng: weight initialization, dropout masks...
    
    torch.cuda.manual_seed_all(SEED)
    # Giải thích:
    # - Set seed cho TẤT CẢ GPU (nếu có nhiều GPU)
    # - Đảm bảo kết quả giống nhau trên mọi GPU

    print(f"🔥 Hardware: {DEVICE} | Workers: {NUM_WORKERS}")
    # In ra thông tin hardware để biết đang train trên gì
    
    # ============================================================
    # PHẦN 3: CHUẨN BỊ DỮ LIỆU
    # Flow: Load → Split → Separate Transforms → DataLoader
    # ============================================================
    print("📂 Đang đọc dữ liệu...")
    
    # --- 3.1. Load Dataset ---
    full_ds = Cv2PreprocessDataset(DATA_PATH, transform=None)
    # Giải thích:
    # - Load toàn bộ dataset KHÔNG có transform
    # - Tại sao None? Vì train/val cần transform KHÁC NHAU
    # - Train: Augmentation (flip, rotate...)
    # - Val: Chỉ resize + normalize (không augment)
    
    # --- 3.2. Train/Val Split ---
    train_size = int(0.8 * len(full_ds))
    # Giải thích:
    # - 80% cho training
    # - int(): Làm tròn xuống (ví dụ: 0.8 * 1000 = 800)
    
    val_size = len(full_ds) - train_size
    # Giải thích:
    # - 20% còn lại cho validation
    # - Dùng phép trừ để đảm bảo train_size + val_size = total
    
    split_gen = torch.Generator().manual_seed(SEED)
    # Giải thích:
    # - Tạo random generator riêng cho việc split
    # - Dùng SEED để đảm bảo mỗi lần chạy, split giống nhau
    # - Quan trọng: Val set phải giữ nguyên để so sánh các lần train
    
    train_indices, val_indices = random_split(
        range(len(full_ds)),
        [train_size, val_size],
        generator=split_gen
    )
    # Giải thích:
    # - random_split(): Chia ngẫu nhiên indices
    # - range(len(full_ds)): [0, 1, 2, ..., 999] (nếu 1000 ảnh)
    # - [train_size, val_size]: [800, 200]
    # - generator: Dùng seed đã set
    # → train_indices: [543, 12, 789, ...]  (800 số)
    # → val_indices:   [45, 234, 678, ...]  (200 số)
    
    # --- 3.3. Create Separate Datasets với Transforms Khác Nhau ---
    train_ds = Cv2PreprocessDataset(DATA_PATH, transform=transform_config['train'])
    # Giải thích:
    # - Tạo dataset cho TRAIN với augmentation
    # - transform_config['train']: flip, rotate, brightness...
    # - Mục đích: Tăng độ đa dạng data → model tổng quát hơn
    
    val_ds = Cv2PreprocessDataset(DATA_PATH, transform=transform_config['val'])
    # Giải thích:
    # - Tạo dataset cho VAL KHÔNG có augmentation
    # - transform_config['val']: chỉ resize + normalize
    # - Tại sao không augment? Muốn đánh giá ĐÚNG khả năng model
    
    train_ds.samples = [full_ds.samples[i] for i in train_indices]
    # Giải thích:
    # - Gán lại samples của train_ds = subset từ full_ds
    # - full_ds.samples: List[(img_path, label), ...]
    # - train_indices: [543, 12, 789, ...]
    # → train_ds.samples = [full_ds.samples[543], full_ds.samples[12], ...]
    
    val_ds.samples = [full_ds.samples[i] for i in val_indices]
    # Tương tự cho val set

    # --- 3.4. Check CUDA availability cho tối ưu DataLoader ---
    use_cuda = (DEVICE == 'cuda')
    # Giải thích:
    # - Biến boolean để check có dùng GPU không
    # - Dùng để config pin_memory, persistent_workers...

    # --- 3.5. Create DataLoaders ---
    train_loader = DataLoader(
        train_ds,
        # Dataset để load
        
        batch_size=BATCH_SIZE,
        # Giải thích:
        # - Mỗi lần yield 32 ảnh (1 batch)
        # - GPU xử lý 32 ảnh song song → hiệu quả
        
        shuffle=True,
        # Giải thích:
        # - Xáo trộn thứ tự ảnh mỗi epoch
        # - Tại sao? Tránh model học "thứ tự" thay vì "nội dung"
        # - Ví dụ: Nếu CR luôn đầu tiên → model bias
        
        num_workers=NUM_WORKERS,
        # Giải thích:
        # - Số tiến trình CPU load data song song
        # - 4 workers → 4 threads chuẩn bị data cho GPU
        # - GPU không phải chờ → utilization cao
        
        persistent_workers=NUM_WORKERS > 0,
        # Giải thích:
        # - True: Giữ workers SỐNG giữa các epochs
        # - False: Hủy và tạo lại workers mỗi epoch (chậm)
        # - Điều kiện: Chỉ bật khi có workers (NUM_WORKERS > 0)
        # - Lợi ích: Tiết kiệm 3-5 giây/epoch
        
        pin_memory=use_cuda,
        # Giải thích:
        # - True: Lock memory vào RAM, transfer GPU nhanh hơn
        # - Cơ chế: Pageable RAM → Pinned RAM → GPU VRAM
        # - Chỉ bật khi có CUDA vì không cần thiết cho CPU
        # - Lợi ích: Transfer nhanh hơn 10-20%
    )
                               
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        
        shuffle=False,
        # Giải thích:
        # - Validation KHÔNG shuffle
        # - Tại sao? Không cần vì không training
        # - Giữ thứ tự giúp debug dễ hơn (biết ảnh nào sai)
        
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=use_cuda,
        # Tương tự train_loader
    )

    print(f"✅ Đã tải: {len(train_ds)} ảnh Train | {len(val_ds)} ảnh Val")
    # In ra số lượng để check split đúng chưa

    # ============================================================
    # PHẦN 4: XÂY DỰNG MODEL
    # Strategy: Partial Freezing + Discriminative Learning Rates
    # ============================================================
    print("🛠️ Đang khởi tạo MobileNetV2...")
    
    model = models.mobilenet_v2(weights='DEFAULT')
    # Giải thích:
    # - Load MobileNetV2 pretrained trên ImageNet
    # - weights='DEFAULT': Dùng weights tốt nhất hiện có
    # - Model đã học: edges, textures, shapes từ 1.2M ảnh ImageNet
    # - Cấu trúc:
    #   * model.features: 17 blocks (convolutional layers)
    #   * model.classifier: 2 layers (avgpool + linear 1280→1000)

    # --- 4.1. Freeze Backbone ---
    for param in model.features.parameters():
        param.requires_grad = False
    # Giải thích:
    # - Đóng băng TẤT CẢ parameters trong features
    # - param.requires_grad = False: Không tính gradient, không update
    # - Tại sao? Features đã học tốt từ ImageNet, không cần train lại
    # - Lợi ích:
    #   * Training nhanh hơn (ít parameters)
    #   * Ít VRAM (ít gradient)
    #   * Tránh overfit (giữ kiến thức tổng quát)

    # --- 4.2. Unfreeze Last N Blocks ---
    if UNFREEZE_LAST_N_BLOCKS and UNFREEZE_LAST_N_BLOCKS > 0:
        # Giải thích điều kiện:
        # - UNFREEZE_LAST_N_BLOCKS: Check not None/not 0
        # - > 0: Check là số dương
        # - Chỉ chạy nếu muốn unfreeze (có thể set 0 để full freeze)
        
        last_blocks = list(model.features.children())[-UNFREEZE_LAST_N_BLOCKS:]
        # Giải thích:
        # - model.features.children(): Iterator qua các sub-modules
        # - list(...): Convert thành list
        # - [-UNFREEZE_LAST_N_BLOCKS:]: Lấy N blocks cuối
        # - Ví dụ: N=2 → lấy features[15], features[16]
        
        for block in last_blocks:
            for param in block.parameters():
                param.requires_grad = True
        # Giải thích:
        # - Lặp qua từng block được chọn
        # - Set requires_grad = True: BẬT lại gradient
        # - Blocks này sẽ được fine-tune với BACKBONE_LR
        # - Tại sao blocks cuối? Học "high-level features" gần task hơn

    # --- 4.3. Replace Classifier Head ---
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        # Giải thích:
        # - Dropout: Tắt ngẫu nhiên 20% neurons mỗi forward pass
        # - Công thức: output = input * mask (mask: 80% là 1, 20% là 0)
        # - Tại sao? Tránh overfitting, model phụ thuộc nhiều neurons
        # - Training: Dropout BẬT, Inference: Dropout TẮT
        
        nn.Linear(model.last_channel, NUM_CLASSES)
        # Giải thích:
        # - Linear: Fully connected layer (y = Wx + b)
        # - model.last_channel: 1280 (output của features)
        # - NUM_CLASSES: 4 (CR, LP, OK, PO)
        # - Shape: [batch, 1280] → [batch, 4]
        # - Layer này LUÔN requires_grad=True (mới tạo, chưa train)
    )
    
    model = model.to(DEVICE)
    # Giải thích:
    # - Chuyển toàn bộ model lên GPU/CPU
    # - Làm 1 LẦN ở đây, không trong loop
    # - Sau này chỉ cần chuyển data: images.to(DEVICE)

    # ============================================================
    # PHẦN 5: CÔNG CỤ HUẤN LUYỆN
    # Loss + Optimizer + Scheduler + AMP
    # ============================================================
    
    # --- 5.1. Loss Function ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # Giải thích:
    # - CrossEntropyLoss: Dùng cho multi-class classification
    # - Công thức: -log(softmax(output)[target_class])
    # - label_smoothing=0.05: Làm mềm nhãn
    #   * Thay vì: [0, 0, 1, 0] (one-hot)
    #   * Thành:   [0.0125, 0.0125, 0.95, 0.0125] (smoothed)
    # - Lợi ích: Model ít "overconfident", tổng quát hơn

    # --- 5.2. Optimizer với Discriminative Learning Rates ---
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    # Giải thích:
    # - Lấy TẤT CẢ parameters trong features CÓ requires_grad=True
    # - List comprehension: [p for p in ... if condition]
    # - Kết quả: Chỉ có parameters của 2 blocks cuối (đã unfreeze)
    
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    # Giải thích:
    # - Lấy parameters của classifier
    # - Tất cả đều requires_grad=True vì mới tạo
    
    optimizer = optim.AdamW(
        # Giải thích AdamW:
        # - Adam: Adaptive Moment Estimation (tự điều chỉnh LR)
        # - W: Weight decay được implement ĐÚNG (khác Adam gốc)
        # - Tốt hơn SGD cho transfer learning
        
        [
            {'params': backbone_params, 'lr': BACKBONE_LR},
            # Giải thích:
            # - Group 1: Backbone parameters
            # - lr: 1e-4 (thấp vì đã train sẵn)
            # - Update nhẹ nhàng, giữ kiến thức cũ
            
            {'params': head_params, 'lr': HEAD_LR},
            # Giải thích:
            # - Group 2: Head parameters
            # - lr: 1e-3 (cao hơn backbone gấp 10)
            # - Update mạnh vì chưa train, cần học nhiều
        ],
        weight_decay=WEIGHT_DECAY,
        # Giải thích:
        # - L2 regularization: loss += weight_decay * ||weights||²
        # - 1e-4: Phạt weights lớn, tránh overfit
        # - Apply cho CẢ 2 groups
    )
    
    # --- 5.3. Learning Rate Scheduler ---
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # Giải thích:
    # - Cosine Annealing: LR giảm theo dạng hình sin
    # - Warm Restarts: Tăng LR lên lại sau mỗi chu kỳ
    # - T_0=5: Chu kỳ đầu tiên 5 epochs
    # - T_mult=2: Mỗi chu kỳ sau dài gấp đôi trước đó
    # - Chu kỳ: [0-5], [5-15], [15-35]
    # - Flow:
    #   Epoch 0: LR = 0.001 (max)
    #   Epoch 2.5: LR = 0.0005 (giảm)
    #   Epoch 5: LR = 0.0001 (min) → RESTART → 0.001
    # - Lợi ích: Restart giúp "nhảy" ra local minimum, tìm solution tốt hơn
    
    # --- 5.4. Mixed Precision Scaler ---
    scaler = GradScaler(enabled=use_cuda)
    # Giải thích:
    # - GradScaler: Scale gradients cho Mixed Precision Training
    # - enabled=use_cuda: Chỉ bật khi có GPU (CPU không hỗ trợ FP16)
    # - Vấn đề FP16: Gradient quá nhỏ → underflow → 0
    # - Giải pháp: Nhân gradient lên (scale) → backward → chia xuống → update
    # - Ví dụ:
    #   * Gradient thật: 0.00001
    #   * Scale lên: 0.00001 × 65536 = 0.65536
    #   * Backward không bị underflow
    #   * Scale xuống: 0.65536 / 65536 = 0.00001
    #   * Update weights với giá trị đúng

    # --- 5.5. Early Stopping Config ---
    best_acc = 0.0
    # Giải thích:
    # - Lưu accuracy tốt nhất từ trước đến giờ
    # - Khởi tạo 0.0, sẽ update khi val_acc > best_acc
    
    patience = 5
    # Giải thích:
    # - Số epochs "chịu đựng" khi không tiến bộ
    # - Nếu 5 epochs liên tiếp không cải thiện → DỪNG
    
    bad_epochs = 0
    # Giải thích:
    # - Đếm số epochs liên tiếp không tiến bộ
    # - Reset về 0 khi có cải thiện
    # - Tăng lên 1 khi không cải thiện
    # - Dừng khi bad_epochs >= patience

    # ============================================================
    # PHẦN 6: VÒNG LẶP HUẤN LUYỆN CHÍNH
    # Flow: Train → Validate → Update LR → Save if best → Early Stop
    # ============================================================
    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN (ĐA LUỒNG)...")
    
    for epoch in range(NUM_EPOCHS):
        # Giải thích:
        # - Lặp qua NUM_EPOCHS lần (tối đa 50)
        # - Mỗi epoch: model nhìn toàn bộ training set 1 lần
        # - Có thể dừng sớm nếu bad_epochs >= patience
        
        start_time = time.time()
        # Đo thời gian để biết mỗi epoch mất bao lâu
        
        # ============================================================
        # PHẦN 6.1: TRAINING PHASE
        # ============================================================
        model.train()
        # Giải thích:
        # - Chuyển model sang training mode
        # - Ảnh hưởng:
        #   * Dropout: BẬT (tắt 20% neurons ngẫu nhiên)
        #   * BatchNorm: Cập nhật running statistics
        #   * Gradient: Được tính toán
        
        # --- Khởi tạo metrics ---
        train_loss = 0
        # Giải thích:
        # - Tổng loss của toàn bộ training set
        # - Sẽ tính trung bình sau: train_loss / train_total
        
        train_correct = 0
        # Giải thích:
        # - Số ảnh dự đoán ĐÚNG
        # - Dùng để tính accuracy: train_correct / train_total
        
        train_total = 0
        # Giải thích:
        # - Tổng số ảnh đã xử lý
        # - Bằng len(train_ds) sau khi hết epoch
        
        # --- Progress Bar ---
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=True)
        # Giải thích:
        # - tqdm: Tạo progress bar đẹp
        # - train_loader: Iterable để loop qua
        # - desc: Mô tả hiển thị ở đầu bar
        # - leave=True: Giữ lại bar sau khi xong (xem lịch sử)
        # - Output: Epoch [1/50]: 100%|████| 125/125 [00:15<00:00, 8.1it/s]
        
        for images, labels in loop:
            # Giải thích:
            # - Loop qua từng batch trong train_loader
            # - images: [batch_size, 3, 224, 224] (32 ảnh RGB)
            # - labels: [batch_size] (32 nhãn: 0-3 cho CR, LP, OK, PO)
            
            # --- Chuyển data lên GPU ---
            images = images.to(DEVICE, non_blocking=use_cuda)
            labels = labels.to(DEVICE, non_blocking=use_cuda)
            # Giải thích:
            # - .to(DEVICE): Chuyển tensor từ CPU RAM → GPU VRAM
            # - non_blocking=True: Async transfer (CPU tiếp tục chạy)
            # - Chỉ bật non_blocking khi có CUDA
            # - Flow: CPU chuẩn bị batch kế → GPU xử lý batch hiện tại
            # - Lợi ích: Giảm idle time
            
            # --- Xóa gradient cũ ---
            optimizer.zero_grad()
            # Giải thích:
            # - PyTorch GÂY DỒN gradient mặc định
            # - Batch 1: grad = [0.5, 0.3]
            # - Batch 2: grad = [0.5, 0.3] + [0.2, 0.1] = [0.7, 0.4] ← SAI!
            # - Phải xóa trước mỗi batch:
            # - Batch 1: grad = [0.5, 0.3]
            # - zero_grad() → grad = [0, 0]
            # - Batch 2: grad = [0.2, 0.1] ← ĐÚNG!
            
            # --- Forward Pass với Mixed Precision ---
            amp_ctx = autocast(device_type='cuda', dtype=torch.float16, enabled=use_cuda) if use_cuda else nullcontext()
            # Giải thích:
            # - Tạo context manager cho Mixed Precision
            # - use_cuda=True:
            #   * autocast: Tự động chọn FP16/FP32 cho từng op
            #   * device_type='cuda': Chỉ định GPU
            #   * dtype=torch.float16: Precision mặc định
            # - use_cuda=False:
            #   * nullcontext(): Context manager "rỗng", không làm gì
            #   * CPU không hỗ trợ FP16 → dùng FP32 bình thường
            
            with amp_ctx:
                # Giải thích with statement:
                # - Vào context: autocast bật, các ops bên trong dùng FP16
                # - Ra context: autocast tắt, ops ngoài dùng FP32
                # - Tự động cleanup khi xong hoặc có exception
                
                outputs = model(images)
                # Giải thích:
                # - Forward pass: đưa images qua model
                # - images: [32, 3, 224, 224]
                # - outputs: [32, 4] (32 ảnh, 4 scores cho 4 classes)
                # - Với AMP:
                #   * Convolutions: FP16 (nhanh)
                #   * Matrix multiply: FP16 (nhanh)
                #   * Softmax, loss: FP32 (chính xác)
                # - Flow:
                #   features → [32, 1280]
                #   classifier → [32, 4]
                #   Ví dụ output: [[2.1, -0.5, 1.3, -1.2], ...] (logits)
                
                loss = criterion(outputs, labels)
                # Giải thích:
                # - Tính loss giữa predictions và ground truth
                # - criterion = CrossEntropyLoss
                # - outputs: [32, 4] logits (chưa softmax)
                # - labels: [32] indices (0-3)
                # - Bên trong criterion:
                #   1. Softmax: logits → probabilities
                #      [2.1, -0.5, 1.3, -1.2] → [0.65, 0.05, 0.25, 0.02]
                #   2. Log: -log(prob[correct_class])
                #      Label=0 (CR) → -log(0.65) = 0.43
                #   3. Average: mean(losses)
                # - Label smoothing: Làm mềm [0,0,1,0] → [0.0125,0.0125,0.95,0.0125]
                # - Output: scalar loss (ví dụ: 0.543)
            
            # --- Backward Pass với Gradient Scaling ---
            if use_cuda:
                # Giải thích điều kiện:
                # - Chỉ dùng GradScaler khi có GPU
                # - CPU: backward bình thường
                
                scaler.scale(loss).backward()
                # Giải thích từng bước:
                # 1. scaler.scale(loss):
                #    - Nhân loss lên: loss * scale_factor (ví dụ: × 65536)
                #    - Loss gốc: 0.543
                #    - Scaled: 0.543 × 65536 = 35585.088
                #    - Tại sao? Tránh gradient quá nhỏ (underflow)
                # 
                # 2. .backward():
                #    - Tính gradient ngược từ loss về tất cả weights
                #    - Chain rule: ∂loss/∂w = ∂loss/∂output × ∂output/∂w
                #    - Lưu gradient vào param.grad của mỗi parameter
                #    - Gradient cũng bị scale lên × 65536
                #    - Ví dụ:
                #      * Gradient thật: 0.00001
                #      * Scaled gradient: 0.00001 × 65536 = 0.65536
                #      * Không bị underflow (thành 0) trong FP16
                
                scaler.step(optimizer)
                # Giải thích:
                # 1. Unscale gradients:
                #    - Chia gradient xuống: grad / scale_factor
                #    - 0.65536 / 65536 = 0.00001 (gradient thật)
                # 
                # 2. Check for inf/nan:
                #    - Nếu có: Skip update (gradient explosion)
                #    - Giảm scale_factor cho lần sau
                # 
                # 3. optimizer.step():
                #    - Cập nhật weights: w_new = w_old - lr × grad
                #    - AdamW thực tế phức tạp hơn (momentum, adaptive lr...)
                #    - Backbone: lr = 1e-4
                #    - Head: lr = 1e-3
                #    - Ví dụ:
                #      * w_old = 0.5
                #      * grad = 0.00001
                #      * lr = 0.001
                #      * w_new = 0.5 - 0.001 × 0.00001 = 0.49999999
                
                scaler.update()
                # Giải thích:
                # - Cập nhật scale_factor cho lần sau
                # - Nếu không có inf/nan nhiều lần → tăng scale_factor
                # - Nếu có inf/nan → giảm scale_factor
                # - Dynamic scaling: Tự động điều chỉnh optimal scale
                # - Mục đích: Maximize precision, minimize underflow
                
            else:
                # Giải thích branch này:
                # - CPU không hỗ trợ FP16
                # - Backward & update bình thường
                
                loss.backward()
                # Giải thích:
                # - Tính gradient như trên nhưng không scale
                # - FP32 có range lớn → không cần scale
                
                optimizer.step()
                # Giải thích:
                # - Update weights trực tiếp
                # - Không cần unscale vì không scale
            
            # --- Tính Metrics cho Batch này ---
            batch_loss = loss.item()
            # Giải thích:
            # - .item(): Chuyển tensor scalar → Python number
            # - Tensor: tensor(0.543, device='cuda:0') → 0.543
            # - Tại sao? Tính toán metrics trên CPU, tiết kiệm VRAM
            
            train_loss += batch_loss * images.size(0)
            # Giải thích:
            # - Cộng dồn loss (có trọng số batch size)
            # - batch_loss: Loss trung bình của 32 ảnh
            # - images.size(0): 32 (batch size)
            # - Tại sao nhân? Để tính weighted average sau
            # - Ví dụ:
            #   * Batch 1: 32 ảnh, loss=0.5 → +16
            #   * Batch 2: 32 ảnh, loss=0.6 → +19.2
            #   * ...
            #   * Batch cuối: 16 ảnh, loss=0.4 → +6.4
            #   * Total: 41.6
            #   * Average: 41.6 / 800 (tổng ảnh) = 0.052
            
            _, predicted = torch.max(outputs, 1)
            # Giải thích:
            # - torch.max(outputs, 1): Tìm max theo dimension 1 (classes)
            # - outputs: [32, 4]
            # - Trả về: (values, indices)
            # - Ví dụ:
            #   output[0] = [2.1, -0.5, 1.3, -1.2]
            #   max value = 2.1, index = 0
            #   predicted[0] = 0 (dự đoán class CR)
            # - _: Bỏ qua values, chỉ lấy indices
            # - predicted: [32] tensor chứa class dự đoán (0-3)
            
            train_total += labels.size(0)
            # Giải thích:
            # - Đếm tổng số ảnh đã xử lý
            # - labels.size(0): 32 (batch size)
            # - Sau epoch: train_total = 800 (số ảnh training)
            
            train_correct += (predicted == labels).sum().item()
            # Giải thích từng phần:
            # 1. (predicted == labels):
            #    - So sánh element-wise
            #    - predicted: [0, 2, 1, 3, ...]
            #    - labels:    [0, 1, 1, 3, ...]
            #    - Result:    [True, False, True, True, ...]
            # 
            # 2. .sum():
            #    - Đếm số True
            #    - True = 1, False = 0
            #    - Ví dụ: [True, False, True, True] → 3
            # 
            # 3. .item():
            #    - Chuyển tensor → Python int
            #    - tensor(3) → 3
            # 
            # - Kết quả: Số ảnh dự đoán đúng trong batch này
            
            loop.set_postfix(loss=batch_loss, acc=train_correct/train_total)
            # Giải thích:
            # - Cập nhật thông tin hiển thị ở cuối progress bar
            # - loss: Loss của batch hiện tại
            # - acc: Accuracy tích lũy từ đầu epoch
            # - Output: [...loss=0.543, acc=0.78]
            # - Real-time monitoring: Nhìn ngay biết training có ổn không
            
        # --- Tính Metrics Tổng cho Toàn Bộ Training Set ---
        train_acc = train_correct / train_total
        # Giải thích:
        # - Accuracy = Số đúng / Tổng số
        # - Ví dụ: 650 / 800 = 0.8125 = 81.25%
        # - Đây là accuracy trên TRAINING set
        
        train_loss_avg = train_loss / train_total
        # Giải thích:
        # - Loss trung bình = Tổng loss / Tổng số ảnh
        # - Weighted average vì batch cuối có thể nhỏ hơn
        # - Ví dụ: 41.6 / 800 = 0.052
        
        # ============================================================
        # PHẦN 6.2: VALIDATION PHASE
        # ============================================================
        model.eval()
        # Giải thích:
        # - Chuyển sang evaluation mode
        # - Ảnh hưởng:
        #   * Dropout: TẮT (dùng 100% neurons)
        #   * BatchNorm: Dùng running stats đã lưu (không update)
        #   * Gradient: Không tính (trong torch.no_grad())
        # - Tại sao cần? Muốn đánh giá ĐÚNG khả năng của model
        
        # --- Khởi tạo metrics ---
        val_loss = 0
        val_correct = 0
        val_total = 0
        # Tương tự train metrics
        
        with torch.no_grad():
            # Giải thích:
            # - Tắt gradient computation
            # - Tại sao?
            #   * Validation không cần gradient (không update weights)
            #   * Tiết kiệm memory: Không lưu computation graph
            #   * Nhanh hơn: Không tính toán gradient
            # - Lợi ích:
            #   * Memory giảm 50%
            #   * Tốc độ tăng 30%
            # - Cơ chế:
            #   * Tensor.requires_grad = False temporarily
            #   * Không build computation graph
            
            for images, labels in val_loader:
                # Giải thích:
                # - Loop qua validation set
                # - Không có progress bar (val nhanh hơn train)
                # - Không shuffle (thứ tự cố định)
                
                images = images.to(DEVICE, non_blocking=use_cuda)
                labels = labels.to(DEVICE, non_blocking=use_cuda)
                # Chuyển data lên GPU, tương tự train
                
                outputs = model(images)
                # Giải thích:
                # - Forward pass KHÔNG có autocast
                # - Tại sao? Đã ở ngoài training loop, dùng FP32 đầy đủ
                # - Val cần chính xác tuyệt đối → FP32 tốt hơn
                # - outputs: [32, 4] logits
                
                loss = criterion(outputs, labels)
                # Tính loss, tương tự train
                
                val_loss += loss.item() * images.size(0)
                # Cộng dồn loss (weighted)
                
                _, predicted = torch.max(outputs, 1)
                # Lấy class prediction
                
                val_total += labels.size(0)
                # Đếm tổng số ảnh
                
                val_correct += (predicted == labels).sum().item()
                # Đếm số dự đoán đúng
                
        # --- Tính Metrics Validation ---
        val_acc = val_correct / val_total
        # Giải thích:
        # - Accuracy trên validation set
        # - ĐÂY LÀ METRIC QUAN TRỌNG NHẤT
        # - Đánh giá khả năng tổng quát của model
        # - So sánh:
        #   * train_acc = 90%, val_acc = 85% → OK (generalize tốt)
        #   * train_acc = 95%, val_acc = 70% → OVERFIT (học thuộc)
        #   * train_acc = 60%, val_acc = 58% → UNDERFIT (chưa học đủ)
        
        val_loss_avg = val_loss / val_total
        # Loss trung bình validation
        
        # ============================================================
        # PHẦN 6.3: UPDATE LEARNING RATE
        # ============================================================
        scheduler.step(epoch)
        # Giải thích:
        # - Cập nhật learning rate theo scheduler
        # - CosineAnnealingWarmRestarts:
        #   * Input: epoch number
        #   * Output: Điều chỉnh optimizer.param_groups[i]['lr']
        # - Flow:
        #   Epoch 0: lr_backbone=1e-4, lr_head=1e-3
        #   Epoch 2: lr giảm theo cosine
        #   Epoch 5: RESTART → lr_backbone=1e-4, lr_head=1e-3
        # - Tự động, không cần làm gì thêm
        
        # ============================================================
        # PHẦN 6.4: LOGGING & DISPLAY
        # ============================================================
        print(f"👉 KQ: Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%} (Loss: {val_loss_avg:.4f})")
        # Giải thích format strings:
        # - {train_acc:.1%}: Format percentage, 1 số thập phân
        #   * 0.8125 → 81.2%
        # - {val_loss_avg:.4f}: Format float, 4 số thập phân
        #   * 0.052134 → 0.0521
        # - Output: 👉 KQ: Train Acc: 81.2% | Val Acc: 78.5% (Loss: 0.0521)
        
        # ============================================================
        # PHẦN 6.5: SAVE BEST MODEL
        # ============================================================
        if val_acc > best_acc:
            # Giải thích điều kiện:
            # - Chỉ save khi val_acc TỐT HƠN best_acc
            # - Ví dụ:
            #   * Epoch 1: val_acc=0.75, best_acc=0 → Save, best_acc=0.75
            #   * Epoch 2: val_acc=0.73, best_acc=0.75 → Không save
            #   * Epoch 3: val_acc=0.78, best_acc=0.75 → Save, best_acc=0.78
            
            best_acc = val_acc
            # Giải thích:
            # - Update best_acc với giá trị mới
            # - Dùng để so sánh các epochs sau
            
            torch.save(model.state_dict(), 'best_mobilenet_hybrid.pth')
            # Giải thích:
            # - Lưu model weights vào file
            # - model.state_dict(): Dictionary chứa tất cả parameters
            #   {
            #     'features.0.0.weight': tensor([...]),
            #     'features.0.0.bias': tensor([...]),
            #     ...
            #     'classifier.1.0.weight': tensor([...]),
            #   }
            # - 'best_mobilenet_hybrid.pth': Tên file
            # - .pth: Extension chuẩn cho PyTorch
            # - Chỉ lưu weights, KHÔNG lưu:
            #   * Kiến trúc model (phải define lại khi load)
            #   * Optimizer state
            #   * Training history
            # - Kích thước file: ~14MB (MobileNetV2)
            
            print("💾 Đã lưu KỶ LỤC MỚI!")
            # Thông báo cho user biết
            
            bad_epochs = 0
            # Giải thích:
            # - Reset counter về 0 vì có tiến bộ
            # - Bắt đầu đếm lại từ đầu
            
        else:
            # Giải thích else:
            # - Trường hợp val_acc KHÔNG tốt hơn best_acc
            # - Model không cải thiện
            
            bad_epochs += 1
            # Giải thích:
            # - Tăng counter lên 1
            # - Đánh dấu 1 epoch "thất bại"
            # - Ví dụ:
            #   * Epoch 10: val_acc giảm → bad_epochs = 1
            #   * Epoch 11: val_acc giảm → bad_epochs = 2
            #   * Epoch 12: val_acc giảm → bad_epochs = 3
            #   * Epoch 13: val_acc tăng → bad_epochs = 0 (reset)
            
        # ============================================================
        # PHẦN 6.6: EARLY STOPPING CHECK
        # ============================================================
        if bad_epochs >= patience:
            # Giải thích điều kiện:
            # - patience = 5: Chịu đựng tối đa 5 epochs không tiến bộ
            # - bad_epochs >= 5: Đã 5 epochs liên tiếp không cải thiện
            # - Kết luận: Model đã hội tụ, tiếp tục train = lãng phí thời gian
            
            print(f"⛔ DỪNG SỚM!")
            # Thông báo dừng sớm
            
            break
            # Giải thích:
            # - Thoát khỏi vòng for epoch
            # - Không chạy các epochs còn lại
            # - Ví dụ:
            #   * NUM_EPOCHS = 50
            #   * Epoch 18: bad_epochs = 5
            #   * Break → Dừng ở epoch 18, không chạy 19-50
            # - Lợi ích:
            #   * Tiết kiệm thời gian (32 epochs × 30s = 16 phút)
            #   * Tránh overfit (train thêm không giúp gì)
            #   * Tự động: Không cần babysit

    # ============================================================
    # PHẦN 7: KẾT THÚC TRAINING
    # ============================================================
    print(f"🏁 XONG! Best Acc: {best_acc:.1%}")
    # Giải thích:
    # - In ra accuracy tốt nhất đạt được
    # - best_acc: Giá trị cao nhất trong quá trình training
    # - Ví dụ: 🏁 XONG! Best Acc: 87.3%
    # - Đây là kết quả cuối cùng của model

# ============================================================
# PHẦN 8: SỬ DỤNG MODEL ĐÃ TRAIN
# ============================================================
"""
Sau khi train xong, load model để inference:

# Load model
model = models.mobilenet_v2()
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 4)
)
model.load_state_dict(torch.load('best_mobilenet_hybrid.pth'))
model.eval()
model = model.to('cuda')

# Predict 1 ảnh
from PIL import Image
import torchvision.transforms as transforms

img = Image.open('test.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0).to('cuda')

with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    class_names = ['CR', 'LP', 'OK', 'PO']
    print(f"Predicted: {class_names[predicted.item()]}")
"""

# ============================================================
# TÓM TẮT FLOW TỔNG THỂ
# ============================================================
"""
1. SETUP:
   - Tạo random seed → Reproducible
   - Load data → Split 80/20
   - Separate transforms cho train/val
   
2. MODEL:
   - Load pretrained MobileNetV2
   - Freeze backbone (features)
   - Unfreeze 2 blocks cuối
   - Replace classifier (1000 → 4 classes)
   
3. TRAINING TOOLS:
   - Loss: CrossEntropyLoss + label smoothing
   - Optimizer: AdamW với 2 learning rates
   - Scheduler: CosineAnnealingWarmRestarts
   - AMP: GradScaler cho mixed precision
   
4. TRAINING LOOP (mỗi epoch):
   a. TRAIN:
      - model.train()
      - Loop qua train_loader
      - Forward với AMP
      - Backward với gradient scaling
      - Update weights
      - Tính accuracy
   
   b. VALIDATION:
      - model.eval()
      - torch.no_grad()
      - Forward (FP32)
      - Tính accuracy
   
   c. UPDATE & SAVE:
      - Scheduler.step()
      - If val_acc > best_acc: Save model
      - Else: bad_epochs += 1
   
   d. EARLY STOP:
      - If bad_epochs >= patience: Break
   
5. RESULT:
   - Best model saved tại 'best_mobilenet_hybrid.pth'
   - Best accuracy: ~85-90%
   - Training time: ~5-10 phút (20 epochs)
"""