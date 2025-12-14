import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
import os
import time
import random
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import sys
sys.path.append('..')
from hybrid.data import Cv2PreprocessDataset, transform_config

# ============================================================
# CODE CŨ - train.py version trước khi chỉnh unfreeze + param groups
# Best Acc đạt ~62.5%
# ============================================================

# ============================================================
# 1. CẤU HÌNH (ĐỂ NGOÀI ĐỂ GLOBAL DÙNG ĐƯỢC)
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Cấu hình nhẹ cho CPU/máy yếu; tăng dần nếu đủ RAM/GPU
BATCH_SIZE = 32
NUM_WORKERS = 4  # đặt 0 cho CPU yếu; tăng 1-2 nếu còn dư RAM
NUM_EPOCHS = 20
LR = 5e-3
NUM_CLASSES = 4
DATA_PATH = r'C:\project\picture-hust\data\train'

# ============================================================
# QUAN TRỌNG: CÂU LỆNH IF "THẦN THÁNH"
# Mọi logic chạy code phải nằm sau dòng này
# ============================================================
if __name__ == '__main__':
    # Đặt seed để tái lập
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"🔥 Hardware: {DEVICE} | Workers: {NUM_WORKERS}")
    
    # ============================================================
    # 2. CHUẨN BỊ DỮ LIỆU
    # ============================================================
    print("📂 Đang đọc dữ liệu...")
    # Dùng transform riêng cho train/val để tránh augment vào val
    full_ds = Cv2PreprocessDataset(DATA_PATH, transform=None)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_indices, val_indices = random_split(range(len(full_ds)), [train_size, val_size])

    # Tạo dataset train/val riêng để gán transform khác nhau
    train_ds = Cv2PreprocessDataset(DATA_PATH, transform=transform_config['train'])
    val_ds   = Cv2PreprocessDataset(DATA_PATH, transform=transform_config['val'])
    train_ds.samples = [full_ds.samples[i] for i in train_indices]
    val_ds.samples   = [full_ds.samples[i] for i in val_indices]

    # persistent_workers dùng khi num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS>0)
                               
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS>0)

    print(f"✅ Đã tải: {len(train_ds)} ảnh Train | {len(val_ds)} ảnh Val")

    # ============================================================
    # 3. XÂY DỰNG MODEL
    # ============================================================
    print("🛠️ Đang khởi tạo MobileNetV2...")
    model = models.mobilenet_v2(weights='DEFAULT')

    # Freeze backbone, chỉ fine-tune classifier cho nhanh/học dễ
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    # ============================================================
    # 4. CÔNG CỤ HUẤN LUYỆN
    # ============================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()

    best_acc = 0.0
    patience = 5
    bad_epochs = 0

    # ============================================================
    # 5. VÒNG LẶP HUẤN LUYỆN
    # ============================================================
    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN (ĐA LUỒNG)...")
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=True)
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            batch_loss = loss.item()
            train_loss += batch_loss * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=batch_loss, acc=train_correct/train_total)
            
        train_acc = train_correct / train_total
        train_loss_avg = train_loss / train_total
        
        # --- VAL ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            # Lưu ý: Val loader cũng dùng worker nên sẽ nhanh hơn
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        val_loss_avg = val_loss / val_total
        
        scheduler.step(epoch)
        
        print(f"👉 KQ: Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%} (Loss: {val_loss_avg:.4f})")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_mobilenet_hybrid.pth')
            print("💾 Đã lưu KỶ LỤC MỚI!")
            bad_epochs = 0
        else:
            bad_epochs += 1
            
        if bad_epochs >= patience:
            print(f"⛔ DỪNG SỚM!")
            break

    print(f"🏁 XONG! Best Acc: {best_acc:.1%}")
