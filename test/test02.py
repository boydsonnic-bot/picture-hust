import sys
import os

# Thêm đường dẫn thư mục cha (project root) và thư mục hybrid vào hệ thống
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
hybrid_dir = os.path.join(parent_dir, 'hybrid')
sys.path.append(hybrid_dir)

# Bây giờ mới import được
import torch
from torch.utils.data import DataLoader
from data import Cv2PreprocessDataset, transform_config # Python đã tìm thấy file data.py
# 1. CẤU HÌNH CƠ BẢN
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = r'C:\project\picture-hust\data\train' # <--- Sửa lại đường dẫn nếu cần

print(f"🔥 Đang test trên thiết bị: {DEVICE}")

# 2. CHUẨN BỊ DỮ LIỆU (Phải có cái này mới test được)
print("📂 Đang đọc dữ liệu...")
try:
    full_ds = Cv2PreprocessDataset(DATA_PATH, transform=transform_config)
    
    # Lấy tạm 80% để test (giống file train)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, _ = random_split(full_ds, [train_size, val_size])
    
    print(f"✅ Đã load xong {len(train_ds)} ảnh để test.")

except Exception as e:
    print(f"❌ Lỗi đọc dữ liệu: {e}")
    exit()

# 3. BẮT ĐẦU TEST BATCH SIZE (Đoạn code bạn muốn chạy)
print("\n🚀 BẮT ĐẦU TEST TẢI TRỌNG GPU...")
print("-" * 30)

for bs in [16, 32, 64, 128]: # Thử thêm cả 128 cho máu
    print(f"Testing Batch Size = {bs} ...", end=" ")
    try:
        # Tạo loader tạm
        test_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
        
        # Bốc thử 1 gói
        images, labels = next(iter(test_loader))
        
        # Ném vào GPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        print("✅ OK! (GPU chịu được)")
        
        # Dọn dẹp bộ nhớ ngay để test cái tiếp theo
        del images, labels
        torch.cuda.empty_cache() 
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ QUÁ TẢI! (Tràn bộ nhớ VRAM)")
        else:
            print(f"❌ Lỗi khác: {e}")
            
    except Exception as e:
        print(f"❌ Lỗi lạ: {e}")

print("-" * 30)
print("🏁 Hoàn tất kiểm tra.")