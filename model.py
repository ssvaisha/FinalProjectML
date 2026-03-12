import os
import random
import pandas as pd
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

#like we used seed in midterm
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


if torch.cuda.is_available():
    device = 'cuda'
    print("cuda is available. using gpu.")

else:
    device ='cpu'
    print("cuda not available. using cpu.")

transform = transforms.Compose([
    transforms.ToTensor(),
    v2.Resize((224,224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   
    transforms.GaussianBlur(kernel_size=3),                 
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    transforms.RandomHorizontalFlip(0.15),
    ])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    v2.Resize((224,224)),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

transform_show = transforms.Compose([
    transforms.ToTensor(),
    v2.Resize((224,224)),
])

for cat in ["train", "test", "val"]:   
    dataset = datasets.ImageFolder(f"archive_smallset/{cat}", transform=transform_show)
    class_names = dataset.classes
    
    plt.figure(figsize=(20, 12))
    plot_index = 1
    
    for class_index, class_name in enumerate(class_names):
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_index][:25]
        
        for idx in class_indices:
            img, label = dataset[idx]
            
            # rows = number of classes
            # cols = 25 images per class
            plt.subplot(len(class_names), 25, plot_index)
            plt.imshow(img.permute(1, 2, 0).clamp(0, 1)) 
            #we added .clamp because it forces every pixel to stay in between 0 and 1 
            #because it was showing an error while plotting in matplot when running the code
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
train_dataset = datasets.ImageFolder("archive_smallset/train")

amount_data = {}

for class_type in train_dataset.classes:
    class_directive = os.path.join('archive_smallset/train',class_type)

    count = len([
        folder_name for folder_name in os.listdir(class_directive) 
        if os.path.isfile(os.path.join(class_directive,folder_name))
    ])
    
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
    print("\n----", split_name, "----")
    for inputs, outputs in loader:
        print("inputs shape:", inputs.shape)
        print("outputs:", outputs.tolist())  

        print("first image first 10 pixel values:", inputs[0].flatten()[:10].tolist())
        break 


train_dataset = datasets.ImageFolder("archive_smallset/train", transform=transform)
val_dataset   = datasets.ImageFolder("archive_smallset/val",   transform=transform_eval)
test_dataset  = datasets.ImageFolder("archive_smallset/test",  transform=transform_eval)

# Creating DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        self.dropout = nn.Dropout(0.3)
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
        x = self.dropout(x)

        output = self.fc2(x)

        return output

model = ConvModel().to(device)

NUM_Epoch = 3
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
val_accuracies = []

#training loop
for epoch in range(NUM_Epoch):
    model.train()
    running_train_loss = 0.0
    train_total = 0

    for train_inputs, train_outputs in train_loader:
        train_inputs = train_inputs.to(device)
        train_outputs = train_outputs.to(device)

        train_preds = model(train_inputs)
        loss = loss_fn(train_preds,train_outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * train_inputs.size(0)
        train_total += train_inputs.size(0)

    avg_train_loss = running_train_loss / train_total
    train_losses.append(avg_train_loss) 

    #validation
    model.eval()
    correct = 0
    total = 0
    running_val_loss = 0.0

    with torch.no_grad():
        for val_inputs, val_outputs in val_loader:
            val_inputs = val_inputs.to(device)
            val_outputs = val_outputs.to(device)

            val_preds = model(val_inputs)
            loss = loss_fn(val_preds, val_outputs)
            running_val_loss += loss.item() * val_inputs.size(0)

            max_value, predicted = torch.max(val_preds,1)

            total += val_outputs.size(0)
            correct += (predicted == val_outputs).sum().item()

    val_accuracy = correct/total
    avg_val_loss = running_val_loss / total

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch} Train Loss: {avg_train_loss}")
    print(f"Epoch {epoch} Validation Loss: {avg_val_loss}")
    print(f"Epoch {epoch} Validation Accuracy: {val_accuracy}")

#testing
model.eval()
correct = 0
total = 0
test_loss = 0.0
alltest_labels = []
alltest_preds = []

with torch.no_grad():
    for test_inputs, test_outputs in test_loader:
        test_inputs = test_inputs.to(device)
        test_outputs = test_outputs.to(device)

        test_preds = model(test_inputs)
        loss = loss_fn(test_preds, test_outputs)
        test_loss += loss.item() * test_inputs.size(0)

        max_value, predicted = torch.max(test_preds,1)
        total += test_outputs.size(0)
        correct += (predicted == test_outputs).sum().item()

        alltest_labels.extend(test_outputs.cpu().tolist())
        alltest_preds.extend(predicted.cpu().tolist())

test_accuracy = correct / total
avgtest_loss = test_loss / total
test_f1 = f1_score(alltest_labels, alltest_preds, average="weighted")

print(f"Test Accuracy: {test_accuracy}")
print(f"Test Loss: {avgtest_loss}")
print(f"Weighted F1 Score: {test_f1}")

cm = confusion_matrix(alltest_labels, alltest_preds)
vt = np.array(cm)
print(vt)
classes = ["COVID19","NORMAL","PNEUMONIA","TUBERCULOSIS"]
for i, label in enumerate(classes):
    tp = vt[i,i]
    fp = vt[:,i].sum()-tp
    fn = vt[i,:].sum()-tp

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    print(f"{label}: Precision: {precision} | Recall: {recall}")


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()

epochs = range(NUM_Epoch)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker="o", label="Train Loss")
plt.plot(epochs, val_losses, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Validation Loss")
plt.legend()
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(epochs, val_accuracies, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.show()
