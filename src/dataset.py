import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMG_SIZE

class InvoiceDataset(Dataset):
    def __init__(self, csv_path, images_dir):
        self.data = pd.read_csv(csv_path)
        self.images_dir = images_dir

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = int(row["label"])
        return image, label
