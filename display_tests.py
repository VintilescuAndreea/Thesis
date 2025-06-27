"""import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import RetinopathyDataset_2, data_transforms, root_dir_2, csv_file_2
from models import create_model_vit

import matplotlib.pyplot as plt
import numpy as np

# Parameters
model_path = 'model_classification_vit_dataset_03_10_25.pth'
num_classes = 5
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = create_model_vit(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Define transforms (use validation transforms here)
transform = data_transforms['validation']

# Prepare dataset and dataloader
test_dataset = RetinopathyDataset_2(csv_file=csv_file_2, root_dir=root_dir_2, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Class names (optional, if you have them)
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Inference loop
all_preds = []
all_labels = []

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         outputs = model(inputs)
#         logits = outputs.logits if hasattr(outputs, 'logits') else outputs
#         _, preds = torch.max(logits, 1)

#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# Optional: print classification report
from sklearn.metrics import classification_report, confusion_matrix

# print("Classification Report:")
# print(classification_report(all_labels, all_preds, target_names=class_names))

# Optional: display some images
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])  # Normalize means from torchvision
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis('off')

# Show a few predictions
from torchvision.utils import make_grid

inputs_batch, labels_batch = next(iter(test_loader))
dim_display = 8
inputs_batch = inputs_batch[:dim_display]
labels_batch = labels_batch[:dim_display]
model.eval()
with torch.no_grad():
    inputs_batch = inputs_batch.to(device)
    outputs = model(inputs_batch)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    _, preds = torch.max(logits, 1)

# Plot
plt.figure(figsize=(10, 5))
for i in range(inputs_batch.size(0)):
    ax = plt.subplot(1, dim_display, i + 1)
    imshow(inputs_batch[i].cpu())
    ax.set_title(f'Pred: {class_names[preds[i]]}\nTrue: {class_names[labels_batch[i]]}')
plt.tight_layout()
plt.show()
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import RetinopathyDataset_2, data_transforms, root_dir_2, csv_file_2
from models import create_model_vit

import matplotlib.pyplot as plt
import numpy as np

# Parameters
model_path = 'model_classification_vit_dataset_03_10_25.pth'
num_classes = 5
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = create_model_vit(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Define transforms (use validation transforms here)
transform = data_transforms['validation']

# Prepare dataset and dataloader
test_dataset = RetinopathyDataset_2(csv_file=csv_file_2, root_dir=root_dir_2, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) ## true- sa mi dea mereu alte pozeS

# Class names (optional, if you have them)
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Inference loop
all_preds = []
all_labels = []

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         outputs = model(inputs)
#         logits = outputs.logits if hasattr(outputs, 'logits') else outputs
#         _, preds = torch.max(logits, 1)

#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# Optional: print classification report
from sklearn.metrics import classification_report, confusion_matrix

# print("Classification Report:")
# print(classification_report(all_labels, all_preds, target_names=class_names))

# Optional: display some images
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])  # Normalize means from torchvision
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis('off')

# Show a few predictions
from torchvision.utils import make_grid

inputs_batch, labels_batch = next(iter(test_loader))
dim_display = 4
inputs_batch = inputs_batch[:dim_display]
labels_batch = labels_batch[:dim_display]
model.eval()
with torch.no_grad():
    inputs_batch = inputs_batch.to(device)
    outputs = model(inputs_batch)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    _, preds = torch.max(logits, 1)

# Plot
plt.figure(figsize=(15, 4))
for i in range(inputs_batch.size(0)):
    ax = plt.subplot(1, dim_display, i + 1)
    imshow(
        inputs_batch[i].cpu(),
        title=f'Pred: {class_names[preds[i]]}\n\nTrue: {class_names[labels_batch[i]]}'
    )
    ax.set_title(f'Pred: {class_names[preds[i]]}\n\nTrue: {class_names[labels_batch[i]]}', fontsize=12, pad=10)

plt.subplots_adjust(wspace=0.6, hspace=0.6)
plt.show()

