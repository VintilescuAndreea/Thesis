import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from models import create_model_vit
from dataset import data_transforms


image_path = 'all_images/4b5ffea77373.png'
model_path = 'model_classification_vit_dataset_03_10_25.pth'
num_classes = 5
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = create_model_vit(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


img = Image.open(image_path).convert('RGB')
transform = data_transforms['validation']  # or 'train' if needed
input_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension


with torch.no_grad():
    outputs = model(input_tensor)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    _, predicted = torch.max(logits, 1)
    predicted_class = class_names[predicted.item()]

plt.imshow(img)
plt.title(f'Predicted: {predicted_class}')
plt.axis('off')
plt.show()


