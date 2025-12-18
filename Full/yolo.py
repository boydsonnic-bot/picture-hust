import torch
from ultralytics import YOLO

device = 0 if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    model = YOLO('yolov8s.pt')  

    model.train(
        data = 'C:\\project\\picture-hust\\auto_labels\\data\\datasetV2.yaml',
        epochs = 30,
        batch = 16,
        imgsz = 320,
        device = device,
        project = 'C:\\project\\picture-hust\\Full\\experiments\\yolo_results',
        name= 'xray_320s16V2'
        )

    val = model.val()
    print("mAP@0.5:", val.box.map50)
