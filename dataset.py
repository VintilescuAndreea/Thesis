import os
import csv
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, random_split


classification_root = ''
images_dir = ''
csv_file_2 = 'train.csv'
root_dir_2 = 'all_images'

# === Transformări standard ===
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ]),
}

# === Dataset KAGGLE (RetinopathyDataset_2) ===
class RetinopathyDataset_2(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_base = self.labels_df.iloc[idx, 0]
        label = int(self.labels_df.iloc[idx, 1])

        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = os.path.join(self.root_dir, img_base + ext)
            if os.path.exists(img_path):
                break
        else:
            raise FileNotFoundError(f"Imaginea {img_base} nu a fost găsită în {self.root_dir}")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# === Dataset IDRiD (RetinopathyDataset) – nu l mai folosesc ===
class RetinopathyDataset(Dataset):
    def __init__(self, image_names, labels, image_dir, transform=None):
        self.image_names = image_names
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def generate_grading_dataset():

    dataset = RetinopathyDataset_2(
        csv_file=csv_file_2,
        root_dir=root_dir_2,
        transform=data_transforms['train']
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    test_dataset.dataset.transform = data_transforms['validation']

    return train_dataset, test_dataset, [], []

if __name__ == "__main__":
    full_dataset = RetinopathyDataset_2(csv_file=csv_file_2,
                                        root_dir=root_dir_2,
                                        transform=data_transforms['train'])

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print("Exemplu din train_dataset:", train_dataset[0])
