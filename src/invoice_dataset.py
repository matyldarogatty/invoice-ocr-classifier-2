import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_SIZE
from layout_features import active_layout_columns, validate_layout_dataframe_columns


class InvoiceDataset(Dataset):
    """Image line crops with integer `label` (optional layout features for CNN+layout)."""

    def __init__(
        self,
        csv_path=None,
        images_dir=None,
        dataframe=None,
        strict=True,
        use_layout_features: bool = False,
        exclude_line_no_feature: bool = False,
    ):
        if dataframe is not None:
            self.data = dataframe.reset_index(drop=True).copy()
        elif csv_path is not None:
            self.data = pd.read_csv(csv_path)
            self.data.columns = [str(c).strip() for c in self.data.columns]
        else:
            raise ValueError("InvoiceDataset requires csv_path or dataframe")
        self.images_dir = images_dir
        self.use_layout_features = use_layout_features
        self.layout_cols = active_layout_columns(exclude_line_no=exclude_line_no_feature)
        if strict:
            for col in ("filename", "label"):
                if col not in self.data.columns:
                    src = csv_path if csv_path else "dataframe"
                    raise ValueError(f"{src!r} must contain column: {col!r}")
            if self.data["label"].isna().any():
                raise ValueError("Empty values in 'label' column are not allowed.")
        if use_layout_features:
            validate_layout_dataframe_columns(
                self.data.columns,
                exclude_line_no=exclude_line_no_feature,
            )
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
        if not self.use_layout_features:
            return image, label
        layout = torch.tensor(
            [float(row[c]) for c in self.layout_cols],
            dtype=torch.float32,
        )
        return image, layout, label
