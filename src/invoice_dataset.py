import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMG_SIZE


class InvoiceDataset(Dataset):
    """Image line crops with integer `label` (same rows can pair with text in a future hybrid model)."""

    def __init__(self, csv_path=None, images_dir=None, dataframe=None, strict=True):
        if dataframe is not None:
            self.data = dataframe.reset_index(drop=True).copy()
        elif csv_path is not None:
            self.data = pd.read_csv(csv_path)
            self.data.columns = [str(c).strip() for c in self.data.columns]
        else:
            raise ValueError("InvoiceDataset requires csv_path or dataframe")
        self.images_dir = images_dir
        if strict:
            for col in ("filename", "label"):
                if col not in self.data.columns:
                    src = csv_path if csv_path else "dataframe"
                    raise ValueError(f"{src!r} must contain column: {col!r}")
            if self.data["label"].isna().any():
                raise ValueError("Empty values in 'label' column are not allowed.")
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(row["label"])
        return image, label
