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

transform = transforms.Compose([transforms.ToTensor(),v2.Resize((224,224)),
                        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                        transforms.RandomHorizontalFlip(0.15),
                        ])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    v2.Resize((224,224)),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])


for cat in ["train", "test", "val"]:   
    dataset = datasets.ImageFolder(f"archive/{cat}", transform=transform)
    class_names = dataset.classes
    
    plt.figure(figsize=(20, 12))
    plot_index = 1
    
    for class_index, class_name in enumerate(class_names):
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_index][:25]
        
        for idx in class_indices:
            img, label = dataset[idx]
            
            # rows = number of classes
            # cols = 10 images per class
            plt.subplot(len(class_names), 25, plot_index)
            plt.imshow(img.permute(1, 2, 0))
            plt.title(cat)
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

amount_data = {}
for class_type in train_dataset.classes:
    class_directive = os.path.join('archive/train',class_type)
    count = len([folder_name for folder_name in os.listdir(class_directive) if os.path.isfile(os.path.join(class_directive,folder_name))])
    
    amount_data[str(class_type)] = count



print(amount_data)

fig, ax = plt.subplots()
names = amount_data.keys()
nums = amount_data.values()
ax.bar(range(len(names)),nums, tick_label = names)
ax.set_ylabel("number of images")
ax.set_xlabel("type of data")


plt.show()



# Looping over each DataLoader
def print_one_batch(loader, split_name):
    print("\n-----", split_name, "-----")
    for inputs, outputs in loader:
        print("inputs shape:", inputs.shape)
        print("outputs:", outputs.tolist())  

        print("first image first 10 pixel values:", inputs[0].flatten()[:10].tolist())
        break 



class MyData(Dataset):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.length = len(inputs)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self,idx):
        return self.inputs[idx], self.outputs[idx]


train_dataset = MyData(datasets.ImageFolder("archive/train", transform=transform))
val_dataset   = MyData(datasets.ImageFolder("archive/val",   transform=transform_eval))
test_dataset  = MyData(datasets.ImageFolder("archive/test",  transform=transform_eval))

# Creating DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

print_one_batch(train_loader, "TRAIN")
print_one_batch(val_loader, "VAL")
print_one_batch(test_loader, "TEST")

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1,1)
        self.conv2 = nn.Conv2d(6,16,3,1,1)
        self.conv3 = nn.Conv2d(16,48,3,1,1)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(37632,1000)
        self.fc2 = nn.Linear(1000,4)
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(start_dim =1)
        x = self.relu(self.fc1(x))
        output = self.fc2(x)
        return output

model = ConvModel()
model.train()
NUM_Epoch = 100
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(NUM_Epoch):
    for train_inputs, train_outputs in train_loader:
        model.train()
        train_preds = model(train_inputs)
        loss = loss_fn(train_preds,train_outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Training Epoch: {epoch}")

