import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# =========================================================
# 1. CẤU HÌNH & SETUP
# =========================================================
st.set_page_config(page_title="HUST AI Inspector", page_icon="🏭", layout="wide")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'best_mobilenet_hybrid.pth'  # Đảm bảo file này nằm cùng thư mục

# ⚠️ QUAN TRỌNG: MODEL CỦA BẠN SẮP XẾP LỚP THEO THỨ TỰ FOLDER
# Bạn phải kiểm tra folder data/train để điền đúng thứ tự vào đây.
# Ví dụ folder là: 0_ok, 1_crack, 2_porosity, 3_undercut
# Thì map sẽ là: 0: 'OK', 1: 'CR', 2: 'PO', 3: 'LP'
CLASS_MAP = {
    0: 'CR',  # Crack (Nứt)
    1: 'LP',  # Lack of Penetration / Undercut
    2: 'OK',  # Đạt
    3: 'PO'   # Porosity (Rỗ khí)
}

# =========================================================
# 2. LOAD MODEL THẬT (MobileNetV2)
# =========================================================
@st.cache_resource
def load_model():
    try:
        # Tái tạo kiến trúc
        model = models.mobilenet_v2(weights=None)
        # Sửa đầu ra thành 4 lớp
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, 4) 
        )
        # Load trọng số đã train
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        return None

# =========================================================
# 3. XỬ LÝ ẢNH (PREPROCESS)
# =========================================================
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224), # Cắt giữa giống lúc train
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# =========================================================
# 4. GIAO DIỆN CHÍNH
# =========================================================
def main():
    st.sidebar.title("🎛️ Control Panel")
    mode = st.sidebar.radio("Chọn Chế Độ:", ["Hybrid System (Model Thật)"])
    
    st.title("🏭 Hệ Thống Kiểm Tra Lỗi Hàn (E2E)")
    
    # Load model ngay khi vào app
    model = load_model()
    if model is None:
        st.error(f"❌ Không tìm thấy file '{MODEL_PATH}'. Hãy copy file model vào đây!")
        return

    # --- GIAO DIỆN HYBRID ---
    col_img, col_sensor = st.columns(2)
    
    with col_img:
        st.write("📷 **Đầu vào Hình ảnh**")
        uploaded_file = st.file_uploader("Upload ảnh hàn...", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, width=350, caption="Ảnh đầu vào")



    st.divider()

    # NÚT CHẠY
    if st.button("🚀 PHÂN TÍCH (RUN MODEL)", type="primary"):
        if not uploaded_file:
            st.warning("⚠️ Vui lòng tải ảnh lên trước!")
        else:
            with st.spinner("Đang chạy MobileNetV2..."):
                # 1. Xử lý ảnh
                img_tensor = process_image(image)
                
                # 2. Chạy Model thật
                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    
                    # Lấy class có xác suất cao nhất
                    top_prob, top_idx = probs.topk(1)
                    idx = top_idx.item()
                    confidence = top_prob.item() * 100
                    
                    # Map sang mã OK, CR, PO, LP
                    res_code = CLASS_MAP.get(idx, "UNKNOWN")

                # 3. Hiển thị kết quả
                time.sleep(0.5) # Delay tí cho mượt
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("KẾT QUẢ (CODE)", res_code)
                
                with c2:
                    if res_code == "OK":
                        st.success(f"✅ ĐẠT CHUẨN (Độ tin cậy: {confidence:.1f}%)")
                    else:
                        st.error(f"❌ PHÁT HIỆN LỖI: {res_code} (Độ tin cậy: {confidence:.1f}%)")
                        
                # Hiển thị chi tiết xác suất (Optional)
                st.write("📊 **Chi tiết phân lớp:**")
                st.progress(int(confidence))

if __name__ == "__main__":
    main()