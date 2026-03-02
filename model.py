import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from PIL import Image 
import os

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((100,100)),transforms.ToTensor()])

for split in ["train", "test", "val"]:   
    dataset = datasets.ImageFolder(f"archive/{split}", transform=transform)
    class_names = dataset.classes
    
    plt.figure(figsize=(20, 12))
    plot_index = 1
    
    for class_idx, class_name in enumerate(class_names):
        # Get first 10 images for this class
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx][:10]
        
        for idx in class_indices:
            img, label = dataset[idx]
            
            # rows = number of classes
            # cols = 10 images per class
            plt.subplot(len(class_names), 10, plot_index)
            plt.imshow(img.permute(1, 2, 0))
            plt.title(split)
            #this is to label the y values so its like a bar chart
            if plot_index % 10 == 1:
                plt.ylabel(class_name, fontsize=12)
            
            plt.xticks([])
            plt.yticks([])
            
            plot_index += 1
    
    plt.tight_layout()
    plt.show()

#data analysis
train_dataset = datasets.ImageFolder("archive/train")
test_dataset = datasets.ImageFolder("archive/test")
val_dataset= datasets.ImageFolder("archive/val")
class_types = ['COVID19', 'NORMAL','PNEUMONIA','TURBERCULOSIS']

amount_data = {}
for class_name in train_dataset.classes:
    class_dir = os.path.join('archive/train',class_name)
    count = len([folder_name for folder_name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir,folder_name))])
    
    amount_data[str(class_name)] = count



print(amount_data)

fig, ax = plt.subplots()
names = amount_data.keys()
nums = amount_data.values()
ax.bar(range(len(names)),nums, tick_label = names)
ax.set_ylabel("number of images")
ax.set_xlabel("type of data")


plt.show()