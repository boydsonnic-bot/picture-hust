import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import time

# =========================================================
# 1. CẤU HÌNH & SETUP
# =========================================================
st.set_page_config(page_title="HUST AI Inspector", page_icon="🏭", layout="wide")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH_DEFAULT = 'best_mobilenet_hybrid.pth'  # Đảm bảo file này nằm cùng thư mục

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
def load_model(model_path: str, backbone: str = 'MobileNetV2'):
    try:
        # Tái tạo kiến trúc theo chọn lựa
        if backbone == 'MobileNetV2':
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.last_channel, 4)
            )
            target_layer = 'features[-1]'
        elif backbone == 'ResNet18':
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, 4)
            )
            target_layer = 'layer4'
        else:
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.last_channel, 4)
            )
            target_layer = 'features[-1]'

        # Load trọng số đã train
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return model, target_layer
    except FileNotFoundError:
        return None, None

# =========================================================
# 3. XỬ LÝ ẢNH (PREPROCESS)
# =========================================================
@st.cache_data(show_spinner=False)
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224), # Cắt giữa giống lúc train
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def tensor_to_probs(output: torch.Tensor):
    probs = torch.nn.functional.softmax(output, dim=1)
    return probs.squeeze(0).detach().cpu().numpy()

def make_probs_df(class_map: dict, probs_np: np.ndarray):
    labels = [class_map[i] for i in range(len(probs_np))]
    return pd.DataFrame({'class': labels, 'prob': probs_np})

def apply_threshold(idx: int, conf: float, threshold: float):
    if conf < threshold:
        return 'UNKNOWN', conf
    return CLASS_MAP.get(idx, 'UNKNOWN'), conf

def compute_gradcam(model: nn.Module, input_tensor: torch.Tensor, target_layer_name: str):
    model.zero_grad()

    activations = []
    gradients = []

    # Lấy module theo tên đơn giản
    layer = None
    try:
        if target_layer_name == 'features[-1]':
            layer = model.features[-1]
        elif target_layer_name == 'layer4':
            layer = model.layer4
    except Exception:
        layer = None

    if layer is None:
        return None

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_inp, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    output = model(input_tensor)
    scores = torch.nn.functional.softmax(output, dim=1)
    top_prob, top_idx = scores.topk(1)
    loss = output[:, top_idx.item()].sum()
    loss.backward()

    # Lấy act và grad
    act = activations[-1]  # [B, C, H, W]
    grad = gradients[-1]   # [B, C, H, W]

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # GAP grad
    cam = (weights * act).sum(dim=1, keepdim=True)        # [B,1,H,W]
    cam = torch.relu(cam)

    cam_np = cam.squeeze(0).squeeze(0).cpu().numpy()
    cam_np -= cam_np.min()
    cam_np /= (cam_np.max() + 1e-8)

    h1.remove(); h2.remove()
    return cam_np, top_idx.item(), top_prob.item()

def overlay_cam_on_image(image: Image.Image, cam_np: np.ndarray):
    img_np = np.array(image.convert('RGB'))
    cam_resized = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * cam_resized)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, colored, 0.4, 0)
    return Image.fromarray(overlay)

# =========================================================
# 4. GIAO DIỆN CHÍNH
# =========================================================
def main():
    st.sidebar.title("🎛️ Control Panel")
    mode = st.sidebar.radio("Chọn Chế Độ:", ["Hybrid System (Model Thật)"])
    st.sidebar.markdown(f"**Device:** {DEVICE}")
    backbone = st.sidebar.selectbox("Backbone", ["MobileNetV2", "ResNet18"], index=0)
    model_path = st.sidebar.text_input("Model path", MODEL_PATH_DEFAULT)
    threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    batch_mode = st.sidebar.checkbox("Batch mode (multi images)", value=False)
    use_cam = st.sidebar.checkbox("Use camera input", value=False)
    
    st.title("🏭 Hệ Thống Kiểm Tra Lỗi Hàn ")
    
    # Load model ngay khi vào app
    model, target_layer = load_model(model_path, backbone)
    if model is None:
        st.error(f"❌ Không tìm thấy file '{model_path}'. Hãy copy file model vào đây hoặc sửa đường dẫn!")
        return

    # --- GIAO DIỆN HYBRID ---
    col_img, col_sensor = st.columns(2)

    uploaded_files = None
    image = None
    with col_img:
        st.write("📷 **Đầu vào Hình ảnh**")
        if use_cam:
            cam_file = st.camera_input("Chụp ảnh")
            if cam_file is not None:
                image = Image.open(cam_file).convert('RGB')
                st.image(image, width=350, caption="Ảnh từ camera")
        else:
            uploaded_files = st.file_uploader("Upload ảnh hàn...", type=['jpg', 'png', 'jpeg'], accept_multiple_files=batch_mode)
            if batch_mode and uploaded_files:
                st.write(f"📦 Số ảnh: {len(uploaded_files)}")
            elif not batch_mode and uploaded_files:
                image = Image.open(uploaded_files).convert('RGB')
                st.image(image, width=350, caption="Ảnh đầu vào")



    st.divider()

    # NÚT CHẠY
    if st.button("🚀 PHÂN TÍCH (RUN MODEL)", type="primary"):
        # SINGLE IMAGE MODE
        if not batch_mode:
            if image is None:
                st.warning("⚠️ Vui lòng tải/chụp ảnh trước!")
            else:
                with st.spinner("Đang chạy inference..."):
                    img_tensor = process_image(image)
                    with torch.no_grad():
                        output = model(img_tensor)
                        probs_np = tensor_to_probs(output)
                        top_idx = int(np.argmax(probs_np))
                        top_conf = float(probs_np[top_idx])
                        res_code, conf_used = apply_threshold(top_idx, top_conf, threshold)

                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.metric("KẾT QUẢ (CODE)", res_code)
                        st.metric("CONFIDENCE", f"{conf_used*100:.1f}%")
                    with c2:
                        df = make_probs_df(CLASS_MAP, probs_np)
                        st.bar_chart(df.set_index('class'))

                    # Grad-CAM
                    if target_layer is not None:
                        cam_np, cam_idx, cam_prob = compute_gradcam(model, img_tensor, target_layer)
                        if cam_np is not None:
                            overlay = overlay_cam_on_image(image, cam_np)
                            st.image(overlay, caption=f"Grad-CAM ({CLASS_MAP.get(cam_idx,'UNKNOWN')}: {cam_prob*100:.1f}%)")

        # BATCH MODE
        else:
            if not uploaded_files:
                st.warning("⚠️ Vui lòng upload nhiều ảnh!")
            else:
                results = []
                with st.spinner("Đang xử lý batch..."):
                    for f in uploaded_files:
                        img = Image.open(f).convert('RGB')
                        tensor = process_image(img)
                        with torch.no_grad():
                            output = model(tensor)
                            probs_np = tensor_to_probs(output)
                            top_idx = int(np.argmax(probs_np))
                            top_conf = float(probs_np[top_idx])
                            code, conf_used = apply_threshold(top_idx, top_conf, threshold)
                        results.append({
                            'file': getattr(f, 'name', 'uploaded'),
                            'code': code,
                            'confidence': round(conf_used*100, 2)
                        })
                df_res = pd.DataFrame(results)
                st.dataframe(df_res, use_container_width=True)
                csv_bytes = df_res.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải CSV kết quả", data=csv_bytes, file_name="results.csv", mime="text/csv")

if __name__ == "__main__":
    main()