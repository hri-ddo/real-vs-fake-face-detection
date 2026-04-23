import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from torchvision import transforms
from PIL import Image
import pickle
import io

# --- 1. Re-define the Architecture ---
class DeepCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512) 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. Load the Model Safely to CPU ---
device = torch.device('cpu')

# CRITICAL FIX: You must create the model before loading weights!
model = DeepCNN(num_classes=2)

# CRITICAL FIX: Custom Unpickler to strip CUDA tags from nested pickle files
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        else:
            return super().find_class(module, name)

# Load the file using the custom unpickler
with open("model_data.pkl", "rb") as f:
    checkpoint = CPU_Unpickler(f).load()

# Inject the weights into the model
if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    model.load_state_dict(checkpoint['model_state'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# --- 3. Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. Prediction Function ---
def predict(img):
    img = transform(img).unsqueeze(0).to(device) 
    with torch.no_grad():
        outputs = model(img)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    return {"Fake": float(probabilities[0]), "Real": float(probabilities[1])}

# --- 5. Gradio Interface ---
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Real vs Fake Face Detector",
    description="Upload a face image to see if the CNN thinks it's a real person or AI-generated."
)

if __name__ == "__main__":
    interface.launch()