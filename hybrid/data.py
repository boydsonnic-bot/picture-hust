import cv2
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
from preprocess import pre_process

transforms_list = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
    ])


class CustomDataset(datasets.ImageFolder):

    def __getitem__(self, index):
        path, label = self.samples[index]
        
        img_bgr= cv2.imread(path)
        if img_bgr is None:
            return torch.zeros(3, 224, 224), label
        
        img_gray = pre_process(img_bgr)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)

        if self.transform:
            img_transformed = self.transform(img_pil)
        else:
            img_transformed = transforms.ToTensor()(img_pil)
        return img_transformed, label


# Backwards-compatible exports expected by `train.py`
Cv2PreprocessDataset = CustomDataset
transform_config = transforms_list