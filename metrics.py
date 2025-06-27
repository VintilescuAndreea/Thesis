from models import create_model, create_model_vit
from dataset import generate_grading_dataset, RetinopathyDataset, data_transforms, classification_root, images_dir, csv_file_2, root_dir_2, RetinopathyDataset_2
import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

phase = "validation"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
y_preds = []
label = []


save_load_path = 'model_classification_vit_dataset_03_10_25.pth'
model = create_model_vit(num_classes=5)

#save_load_path = 'model_classification.pth'
#model = create_model(num_classes=5)

model.load_state_dict(torch.load(save_load_path, map_location=torch.device('cpu'), weights_only=True))
model.eval()

model = model.to(device)
criterion = nn.CrossEntropyLoss()


_, _, test_image_names, test_retinopathy_grades = generate_grading_dataset()

full_dataset  = RetinopathyDataset_2(csv_file=csv_file_2,
                                    root_dir=root_dir_2,
                                    transform=data_transforms['train'])

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
from torch.utils.data import random_split
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

'''
test_dataset = RetinopathyDataset(test_image_names, 
                                test_retinopathy_grades, 
                                image_dir=os.path.join(classification_root, images_dir, 'b. Testing Set'), 
                                transform=data_transforms['validation'])
'''
image_datasets = {
        'validation': test_dataset
    }


dataloaders = {
    'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32, shuffle=True)
    }

for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    try:
        loss = criterion(outputs, labels)
    except TypeError:
        logits = outputs.logits
        loss = criterion(logits, labels)

    try:
        _, preds = torch.max(outputs, 1)
    except TypeError:
        _, preds = torch.max(logits, 1)
    y_preds.append(preds.cpu())
    label.append(labels.cpu())

y_pred = np.array(y_preds[0])
y_true = np.array(label[0])

for i in range(1,len(y_preds)):
  y_pred = np.append(y_pred, np.array(y_preds[i]))

for i in range(1,len(label)):
  y_true = np.append(y_true, np.array(label[i]))


from sklearn.metrics import confusion_matrix
cnf = confusion_matrix(y_true, y_pred)


import seaborn as sns


plt.figure(figsize=(8,6), dpi=100)

sns.set(font_scale = 1.1)

ax = sns.heatmap(cnf, annot=True, fmt='d', )


ax.set_xlabel("Predicted Diagnosis", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(['NO_DR', 'MILD', 'MODERATE', 'PROLI', 'SEVERE'])

ax.set_ylabel("Actual Diagnosis", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(['NO_DR', 'MILD', 'MODERATE', 'PROLI', 'SEVERE'])

# set plot title
ax.set_title("Confusion Matrix for the DR Detection Model", fontsize=14, pad=20)

plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn

classes = ['NO_DR', 'MILD', 'MODERATE', 'PROLI', 'SEVERE']

print(classification_report(y_true, y_pred, target_names=classes))