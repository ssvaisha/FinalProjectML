import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
from pathlib import Path

data_folder_path = Path("archive")
images =[]
for split_folder in data_folder_path.iterdir():
    if split_folder.is_dir():
        for class_folder in split_folder.iterdir():
            if class_folder.is_dir():
                image_file = sorted(class_folder.glob("*"))[:10]



    for file_path in image_file:
        print(f"Accessing: {file_path}")
        img = Image.open(file_path)
        images.append(img)

print(f"total images: {len(images)}")
         