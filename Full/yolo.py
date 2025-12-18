from pathlib import Path

LABEL_DIR = Path("data/val/labels")  # đổi nếu cần

for txt_file in LABEL_DIR.glob("*.txt"):
    new_lines = []

    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            cls = parts[0]

            # HOÁN ĐỔI LP <-> PO
            if cls == "1":      # LP nhầm
                cls = "2"
            elif cls == "2":    # PO nhầm
                cls = "1"

            new_lines.append(" ".join([cls] + parts[1:]))

    with open(txt_file, "w") as f:
        f.write("\n".join(new_lines))

print("✅ Đã sửa xong toàn bộ label YOLO")
