import torch
import torch.nn as nn
import timm
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. MODEL ARCHITECTURE ---
class DualTransformerHybrid(nn.Module):
    def __init__(self, num_classes=3):
        super(DualTransformerHybrid, self).__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)

        self.proj_swin = nn.Linear(768, 256)
        self.proj_vit = nn.Linear(192, 256)

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

# --- 2. GRAD-CAM HELPER FUNCTION ---
def reshape_transform(tensor):
    if tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2)
    elif tensor.dim() == 3:
        B, L, C = tensor.size()
        H = int(np.sqrt(L))
        return tensor.reshape(B, H, H, C).permute(0, 3, 1, 2)
    return tensor

# --- 3. INFERENCE & VISUALIZATION PIPELINE ---
def run_demo(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['AD', 'MCI', 'NC']
    
    print(f"Loading model weights from {model_path}...")
    model = DualTransformerHybrid(num_classes=3).to(device)
    # Load your saved weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"Processing image: {image_path}...")
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item() * 100
        prediction_label = class_names[predicted_idx]

    print(f"\n--- DIAGNOSIS ---")
    print(f"Prediction: {prediction_label}")
    print(f"Confidence: {confidence:.2f}%\n")

    # Setup Grad-CAM
    target_layers = [model.swin.layers[-1].blocks[-1].norm1]
    for param in target_layers[0].parameters():
        param.requires_grad = True

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # Prepare image for display
    img_resized = np.array(img.resize((224, 224)))
    img_normalized = np.float32(img_resized) / 255.0
    visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Hybrid Transformer Diagnosis: {prediction_label} ({confidence:.1f}%)", fontsize=16)
    
    plt.subplot(1, 2, 1)
    plt.title("Patient MRI (Skull-Stripped)")
    plt.imshow(img_resized)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Explainability Map")
    plt.imshow(visualization)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    # Point this to a single .png file from your stripped dataset to test
    TEST_IMAGE_PATH = r"D:\Final Year Project\Data\Stripped_Alzheimers_Dataset\MCI\18f\IM00011.png"
    
    # Point this to your saved weights from model
    SAVED_MODEL_PATH = r"best_hybrid_model.pth" 
    
    try:
        run_demo(TEST_IMAGE_PATH, SAVED_MODEL_PATH)
    except FileNotFoundError:
        print("Error: Could not find the image or model file. Please check the paths at the bottom of the script.")