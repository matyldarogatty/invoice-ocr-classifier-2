import sys
import torch
from PIL import Image
from torchvision import transforms
from model import InvoiceCNN
from config import LABELS, IMG_SIZE

device = "cuda" if torch.cuda.is_available() else "cpu"

model = InvoiceCNN().to(device)
model.load_state_dict(torch.load("models/invoice_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

img_path = sys.argv[1]

image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    pred = outputs.argmax(1).item()

print("Obraz:", img_path)
print("Przewidziana klasa:", LABELS[pred])
