# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from timm import create_model  # For Vision Transformer

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data paths
data_dir = 'Lung X-Ray Image'  # Change to your dataset path

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split dataset into training and validation
def split_dataset(data_dir, transform, val_split=0.2):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

train_dataset, val_dataset = split_dataset(data_dir, transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Hybrid model: Vision Transformer + CNN
class HybridViTCNN(nn.Module):
    def __init__(self, num_classes):
        super(HybridViTCNN, self).__init__()
        # Vision Transformer
        self.vit = create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        
        # CNN backbone (ResNet18)
        self.cnn = models.resnet18(pretrained=True)
        cnn_out_features = self.cnn.fc.in_features  # Get the output features of the original fc layer
        self.cnn.fc = nn.Identity()  # Replace the fc layer with Identity

        # Fully connected layer combining ViT and CNN features
        self.fc = nn.Sequential(
            nn.Linear(self.vit.embed_dim + cnn_out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        vit_features = self.vit(x)
        cnn_features = self.cnn(x)
        combined_features = torch.cat((vit_features, cnn_features), dim=1)
        output = self.fc(combined_features)
        return output

# Initialize model
num_classes = len(train_dataset.dataset.classes)
model = HybridViTCNN(num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    print("Training complete")

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

# Save the trained model
torch.save(model.state_dict(), "hybrid_vit_cnn_model.pth")
print("Model saved as 'hybrid_vit_cnn_model.pth'")


# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from timm import create_model  # For Vision Transformer

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data paths
data_dir = 'Lung X-Ray Image'  # Change to your dataset path

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split dataset into training and validation
def split_dataset(data_dir, transform, val_split=0.2):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

train_dataset, val_dataset = split_dataset(data_dir, transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Hybrid model: Vision Transformer + CNN
class HybridViTCNN(nn.Module):
    def __init__(self, num_classes):
        super(HybridViTCNN, self).__init__()
        # Vision Transformer
        self.vit = create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        
        # CNN backbone (ResNet18)
        self.cnn = models.resnet18(pretrained=True)
        cnn_out_features = self.cnn.fc.in_features  # Get the output features of the original fc layer
        self.cnn.fc = nn.Identity()  # Replace the fc layer with Identity

        # Fully connected layer combining ViT and CNN features
        self.fc = nn.Sequential(
            nn.Linear(self.vit.embed_dim + cnn_out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        vit_features = self.vit(x)
        cnn_features = self.cnn(x)
        combined_features = torch.cat((vit_features, cnn_features), dim=1)
        output = self.fc(combined_features)
        return output

# Initialize model
num_classes = len(train_dataset.dataset.classes)
model = HybridViTCNN(num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    print("Training complete")

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

# Save the trained model
torch.save(model.state_dict(), "hybrid_vit_cnn_model.pth")
print("Model saved as 'hybrid_vit_cnn_model.pth'")


# %%
from sklearn.metrics import classification_report

# Function to compute class-wise metrics
def compute_classification_metrics(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    return report

# Generate class-wise metrics
class_names = train_dataset.dataset.classes  # Class names from the dataset
report = compute_classification_metrics(model, val_loader, device, class_names)

# Print metrics in table format
print("Class-Wise Performance Metrics:")
for class_name, metrics in report.items():
    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"{class_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-Score={metrics['f1-score']:.2f}, Support={metrics['support']}")


# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to compute confusion matrix
def plot_confusion_matrix(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Plot confusion matrix
plot_confusion_matrix(model, val_loader, device, class_names)


# %%
import time
import torch

# Measure training time
start_time = time.time()
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)
end_time = time.time()
training_time = end_time - start_time

print(f"Training Time: {training_time / 60:.2f} minutes")

# Measure inference time
model.eval()
with torch.no_grad():
    sample_inputs, _ = next(iter(val_loader))
    sample_inputs = sample_inputs.to(device)
    start_time = time.time()
    _ = model(sample_inputs)
    end_time = time.time()

inference_time = (end_time - start_time) / sample_inputs.size(0)
print(f"Inference Time per Image: {inference_time:.4f} seconds")

# Model size
model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Size: {model_size} parameters")


# %%
from collections import Counter

# Compute class distribution in training and validation sets
train_classes = [label for _, label in train_dataset]
val_classes = [label for _, label in val_dataset]

train_distribution = Counter(train_classes)
val_distribution = Counter(val_classes)

# Print class distribution
print("Training Set Distribution:")
for class_idx, count in train_distribution.items():
    print(f"{class_names[class_idx]}: {count}")

print("Validation Set Distribution:")
for class_idx, count in val_distribution.items():
    print(f"{class_names[class_idx]}: {count}")


# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from torchvision.models import resnet18

# Load pretrained ResNet-18 model
cnn_model = resnet18(pretrained=True)
cnn_model.eval()

# Transformation for input image
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load an example image
image_path = 'Lung X-Ray Image/Viral Pneumonia/1003.jpg'  # Replace with the actual path to your image
input_image = Image.open(image_path).convert('RGB')
input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

# Grad-CAM for feature map visualization
class FeatureExtractor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture gradients
        self.hook = self.target_layer.register_backward_hook(self.save_gradient)
        # Hook to capture activations
        self.hook_act = self.target_layer.register_forward_hook(self.save_activation)

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def save_activation(self, module, input, output):
        self.activations = output

    def __call__(self, x):
        return self.model(x)

    def get_gradients(self):
        return self.gradients

    def get_activations(self):
        return self.activations

# Specify the target layer
target_layer = cnn_model.layer4[-1]  # Last convolutional block
feature_extractor = FeatureExtractor(cnn_model, target_layer)

# Forward pass
output = feature_extractor(input_tensor)

# Backward pass for Grad-CAM
class_idx = output.argmax(dim=1).item()  # Get predicted class index
output[:, class_idx].backward()

# Get gradients and activations
gradients = feature_extractor.get_gradients()
activations = feature_extractor.get_activations()

# Compute Grad-CAM
weights = torch.mean(gradients, dim=[2, 3])  # Global Average Pooling over spatial dimensions
cam = torch.zeros(activations.shape[2:], dtype=torch.float32)

for i, w in enumerate(weights[0]):
    cam += w * activations[0, i, :, :]

# Normalize CAM
cam = torch.relu(cam)
cam = cam / torch.max(cam)
cam = cam.cpu().detach().numpy()

# Resize CAM to match input image size
cam_resized = Resize(input_image.size)(Image.fromarray((cam * 255).astype('uint8')))

# Overlay CAM on input image
plt.figure(figsize=(8, 8))
plt.imshow(input_image, alpha=0.6)
plt.imshow(cam_resized, cmap='jet', alpha=0.4)
plt.axis('off')
plt.title('Feature Map Visualization (Grad-CAM)')
plt.savefig('feature_map_visualization.png')  # Save the visualization
plt.show()


# %%
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Evaluate model on validation data
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification report
print(classification_report(all_labels, all_preds, target_names=train_dataset.dataset.classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.dataset.classes, yticklabels=train_dataset.dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# %%



