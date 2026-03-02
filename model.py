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
from torchvision.transforms import v2

transform = transforms.Compose([transforms.ToTensor(),v2.Resize((100,100)),
                        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                        transforms.RandomHorizontalFlip(0.15),
                        ])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    v2.Resize((100,100)),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])


for split in ["train", "test", "val"]:   
    dataset = datasets.ImageFolder(f"archive/{split}", transform=transform)
    class_names = dataset.classes
    
    plt.figure(figsize=(20, 12))
    plot_index = 1
    
    for class_idx, class_name in enumerate(class_names):
        # Get first 10 images for this class
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx][:25]
        
        for idx in class_indices:
            img, label = dataset[idx]
            
            # rows = number of classes
            # cols = 10 images per class
            plt.subplot(len(class_names), 25, plot_index)
            plt.imshow(img.permute(1, 2, 0))
            plt.title(split)
            #this is to label the y values so its like a bar chart
            if plot_index % 25 == 1:
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

train_dataset = datasets.ImageFolder("archive/train", transform=transform)
val_dataset   = datasets.ImageFolder("archive/val",   transform=transform_eval)
test_dataset  = datasets.ImageFolder("archive/test",  transform=transform_eval)

# Creating DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# Looping over each DataLoader
def print_one_batch(loader, split_name):
    print("\n-----", split_name, "-----")
    for inputs, outputs in loader:
        print("inputs shape:", inputs.shape)
        print("outputs:", outputs.tolist())  

        print("first image first 10 pixel values:", inputs[0].flatten()[:10].tolist())
        break 

print_one_batch(train_loader, "TRAIN")
print_one_batch(val_loader, "VAL")
print_one_batch(test_loader, "TEST")