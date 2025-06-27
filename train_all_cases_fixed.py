import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import Counter
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from dataset import RetinopathyDataset, RetinopathyDataset_2, data_transforms
from model_defs import create_model, create_model_vit


def generate_grading_dataset_fixed():
    base_path = "/content/drive/My Drive/Disertatie/B. Disease Grading"
    csv_folder = "2. Groundtruths"
    train_csv = "a. IDRiD_Disease Grading_Training Labels.csv"
    test_csv = "b. IDRiD_Disease Grading_Testing Labels.csv"

    def read_csv_to_lists(file_path, image_col, grade_col):
        image_names = []
        retinopathy_grades = []
        with open(file_path, mode='r') as file:
            reader = pd.read_csv(file)
            for _, row in reader.iterrows():
                image_names.append(row[image_col] + ".jpg")
                retinopathy_grades.append(int(row[grade_col]))
        return image_names, retinopathy_grades

    train_csv_path = os.path.join(base_path, csv_folder, train_csv)
    test_csv_path = os.path.join(base_path, csv_folder, test_csv)

    train_image_names, train_retinopathy_grades = read_csv_to_lists(train_csv_path, 'Image name', 'Retinopathy grade')
    test_image_names, test_retinopathy_grades = read_csv_to_lists(test_csv_path, 'Image name', 'Retinopathy grade')

    return train_image_names, train_retinopathy_grades, test_image_names, test_retinopathy_grades


def compute_class_weights(dataset, device):
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
    weight_tensor = torch.tensor([weights[i] for i in range(num_classes)], dtype=torch.float32).to(device)
    return weight_tensor


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, save_path='model.pth', patience=5):
    best_loss = float('inf')
    best_model_wts = None
    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'validation']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                        loss = criterion(logits, labels)
                        _, preds = torch.max(logits, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    total_samples += inputs.size(0)

                except Exception as e:
                    print(f"Batch skipped due to error: {e}")
                    continue

            if total_samples == 0:
                print(f"No samples processed in phase '{phase}'. Skipping epoch.")
                continue

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.item())
                scheduler.step()
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.item())
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        model.load_state_dict(best_model_wts)
                        plot_training(train_acc, val_acc, train_loss, val_loss, save_path.replace('.pth', '.png'))
                        return model

    model.load_state_dict(best_model_wts)
    plot_training(train_acc, val_acc, train_loss, val_loss, save_path.replace('.pth', '.png'))
    return model



EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
GAMMA = 0.5
STEP_SIZE = 5


def run_training(dataset_class, model_fn, csv_file=None, root_dir=None, save_path='model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_class == RetinopathyDataset:
        train_image_names, train_grades, test_image_names, test_grades = generate_grading_dataset_fixed()
        train_dir = "/content/drive/My Drive/Disertatie/B. Disease Grading/1. Original Images/a. Training Set"
        test_dir = "/content/drive/My Drive/Disertatie/B. Disease Grading/1. Original Images/b. Testing Set"
        train_dataset = RetinopathyDataset(train_image_names, train_grades, train_dir, data_transforms['train'])
        test_dataset = RetinopathyDataset(test_image_names, test_grades, test_dir, data_transforms['validation'])
    else:
        full_dataset = RetinopathyDataset_2(csv_file, root_dir, transform=data_transforms['train'])
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        torch.manual_seed(42)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        'validation': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    }

    model = model_fn(num_classes=5).to(device)
    params_to_optimize = getattr(model, 'classifier', getattr(model, 'fc', model)).parameters()
    optimizer = optim.AdamW(params_to_optimize, lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    class_weights = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=EPOCHS, save_path=save_path)

# Exemplu rulare:
# run_training(RetinopathyDataset, create_model, save_path="/content/drive/My Drive/Disertatie/resnet_idrid.pth")
