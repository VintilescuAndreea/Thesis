import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim 


from dataset import generate_grading_dataset, RetinopathyDataset, data_transforms, classification_root, images_dir, root_dir_2, csv_file_2, RetinopathyDataset_2
from models import create_model, create_model_vit

def train_model(model, criterion, optimizer, num_epochs=3, save_path='best_model.pth'):
    best_loss = float('inf')  # Initialize best loss to a very high value
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                
                # Ensure correct handling of logits
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = criterion(logits, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                try:
                    _, preds = torch.max(outputs, 1)
                except TypeError:
                    _, preds = torch.max(logits, 1)


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}')

            if phase == "train":
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
            else:
                test_acc.append(epoch_acc)
                test_loss.append(epoch_loss)

                # Save the model if it has the lowest validation loss
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), save_path)
                    print(f'New best model saved with validation loss: {best_loss:.4f}')

    return model




if __name__ =="__main__":
    from collections import Counter

    test_acc , train_acc, test_loss, train_loss = [], [], [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    """
    train_image_names, train_retinopathy_grades, test_image_names, test_retinopathy_grades = generate_grading_dataset()

    
    train_dataset = RetinopathyDataset(train_image_names, 
                                    train_retinopathy_grades, 
                                    image_dir=os.path.join(classification_root, images_dir, 'a. Training Set'), 
                                    transform=data_transforms['train'])

    test_dataset = RetinopathyDataset(test_image_names, 
                                    test_retinopathy_grades, 
                                    image_dir=os.path.join(classification_root, images_dir, 'b. Testing Set'), 
                                    transform=data_transforms['validation'])
    """
    full_dataset  = RetinopathyDataset_2(csv_file=csv_file_2, 
                                    root_dir=root_dir_2,
                                    transform=data_transforms['train'])
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    from torch.utils.data import random_split
    seed = 42
    torch.manual_seed(seed)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    image_datasets = {
        'train': train_dataset,
        'validation': test_dataset
    }

    print("Training set size: ", len(image_datasets['train']))
    print("Validation set size: ", len(image_datasets['validation']))

    dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32, shuffle=True)
    }

    num_classes = 5
    
    def compute_class_weights(dataset):
        labels = [dataset[i][1] for i in range(len(dataset))]  # Assuming dataset[i] returns (image, label)
        class_counts = Counter(labels)
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        # Inverse frequency weighting
        weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
        
        # Convert to tensor and move to device
        weight_tensor = torch.tensor([weights[i] for i in range(num_classes)], dtype=torch.float32).to(device)
        return weight_tensor

    """
    save_load_path = 'model_classification.pth'
    model = create_model(num_classes=num_classes)
    optimizer = optim.Adam(model.fc.parameters())
    """
    save_load_path = 'model_classification_vit_dataset_05_11_25.pth'
    model = create_model_vit(num_classes=num_classes)
    optimizer = optim.Adam(model.classifier.parameters())


    model = model.to(device)
    class_weights = compute_class_weights(train_dataset)

    # Use the weighted loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    num_epochs = 20
    model_trained = train_model(model, criterion, optimizer, num_epochs=num_epochs, save_path=save_load_path)

    te_acc = []

    for i in test_acc :
        te_acc.append(i.cpu())

    tr_acc = []

    for i in train_acc :
        tr_acc.append(i.cpu())

    epochs = [i for i in range(num_epochs)]
    fig, ax = plt.subplots(1, 2)
    train_acc = tr_acc
    train_loss = train_loss
    val_acc = te_acc
    val_loss = test_loss
    fig.set_size_inches(16, 9)

    # Plotting training and validation accuracy
    ax[0].plot(epochs, train_acc, 'b-', marker='o', label='Training Accuracy', linewidth=2)
    ax[0].plot(epochs, val_acc, 'm-', marker='s', label='Validation Accuracy', linewidth=2)
    ax[0].set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax[0].legend(loc="lower right")
    ax[0].set_xlabel("Epochs", fontsize=12)
    ax[0].set_ylabel("Accuracy", fontsize=12)
    ax[0].grid(True)

    # Plotting training and validation loss
    ax[1].plot(epochs, train_loss, 'b-', marker='o', label='Training Loss', linewidth=2)
    ax[1].plot(epochs, val_loss, 'm-', marker='s', label='Validation Loss', linewidth=2)
    ax[1].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel("Epochs", fontsize=12)
    ax[1].set_ylabel("Loss", fontsize=12)
    ax[1].grid(True)

    # Adjust spacing between plots
    plt.tight_layout(pad=3.0)
    plt.show()