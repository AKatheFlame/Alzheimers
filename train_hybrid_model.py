import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. CONFIGURATION ---
# Set dataset path
DATA_DIR = r"D:\Final Year Project\Data\Augmented+orig\AugmentedAlzheimerDataset"
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3 # AD, MCI, NC

# --- 2. DATASET & DATALOADERS ---
print("Preparing dataset...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ImageFolder automatically labels based on folder names (AD=0, MCI=1, NC=2)
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = full_dataset.classes 
print(f"Classes detected: {class_names}")

# 80/20 Train-Validation Split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. DUAL-TRANSFORMER HYBRID MODEL ---
class DualTransformerHybrid(nn.Module):
    def __init__(self, num_classes=3):
        super(DualTransformerHybrid, self).__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)

        self.proj_swin = nn.Linear(768, 256)
        self.proj_vit = nn.Linear(192, 256)

        # Gated Attention for Fusion
        self.attention_weights = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1) 
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat_swin = self.swin(x)
        feat_vit = self.vit(x)

        p_swin = self.proj_swin(feat_swin)
        p_vit = self.proj_vit(feat_vit)

        feature_array = [p_swin.unsqueeze(1), p_vit.unsqueeze(1)]
        stacked_features = torch.cat(feature_array, dim=1) 
        
        attn = self.attention_weights(stacked_features) 
        fused_features = torch.sum(stacked_features * attn, dim=1) 

        return self.classifier(fused_features)

# --- 4. TRAINING LOOP WITH METRIC TRACKING ---
model = DualTransformerHybrid(num_classes=NUM_CLASSES).to(DEVICE)

# Freeze the Swin Transformer
for param in model.swin.parameters():
    param.requires_grad = False

# Freeze the Vision Transformer
for param in model.vit.parameters():
    param.requires_grad = False
print("Transformer backbones frozen. Only training the fusion/classification head.")
# -----------------------------

# Now we only pass the UNfrozen parameters to the optimizer

class_weights = torch.tensor([1.0, 0.7, 2.5]).to(DEVICE) 

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.05)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"Starting training on {DEVICE}...")
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    # Training Phase
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    train_losses.append(running_loss / len(train_loader))
    train_accs.append(100 * correct / total)
    
    # Validation Phase
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(100 * correct / total)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_losses[-1]:.4f}, Acc: {train_accs[-1]:.2f}% | Val Loss: {val_losses[-1]:.4f}, Acc: {val_accs[-1]:.2f}%")
    # Save best model
    current_val_loss = val_losses[-1]
    if current_val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {current_val_loss:.4f}. Saving best model!")
        best_val_loss = current_val_loss
        torch.save(model.state_dict(), 'best_hybrid_model.pth')

torch.save(model.state_dict(), 'alzheimer_hybrid_model.pth')

model.load_state_dict(torch.load('best_hybrid_model.pth'))
# --- 5. EVALUATION & GRAPHS ---
print("\n=== Generating Reports and Graphs ===")

# A. Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# B. Plot Training Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.savefig('training_curves.png')
print("Saved training_curves.png")
plt.close()

# C. Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")
plt.close()

# --- 6. GRAD-CAM EXPLAINABILITY ---
def reshape_transform(tensor):
    # If the tensor is already a grid (Batch, Height, Width, Channels) - Modern timm
    if tensor.dim() == 4:
        # Rearrange to (Batch, Channels, Height, Width) for Grad-CAM
        return tensor.permute(0, 3, 1, 2)
    # If the tensor is flat (Batch, Length, Channels) - Older timm
    elif tensor.dim() == 3:
        B, L, C = tensor.size()
        H = int(np.sqrt(L))
        return tensor.reshape(B, H, H, C).permute(0, 3, 1, 2)
    return tensor

print("\nGenerating Grad-CAM Visualization...")
model.eval()

# Grab one batch from validation set
sample_images, sample_labels = next(iter(val_loader))
img_tensor = sample_images[0:1].to(DEVICE) # Take first image
true_label = class_names[sample_labels[0].item()]

# Target the final layer of the Swin Transformer branch
target_layers = [model.swin.layers[-1].blocks[-1].norm1]

# We must temporarily allow gradients in the target layer so Grad-CAM can do its math
for param in target_layers[0].parameters():
    param.requires_grad = True

cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]

# De-normalize image for visualization
img_rgb = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_rgb = std * img_rgb + mean
img_rgb = np.clip(img_rgb, 0, 1)

# Overlay heatmap
visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

# Plot and save Grad-CAM
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title(f"Original Image (True: {true_label})")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(visualization)
plt.axis('off')
plt.savefig('gradcam_output.png')
print("Saved gradcam_output.png")
plt.show()

print("\nPipeline Complete! All outputs have been saved to your project folder.")