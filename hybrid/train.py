import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
import os
import time

from data import Cv2PreprocessDataset, transform_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
