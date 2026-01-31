import os
import json
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from invoice_dataset import InvoiceDataset
from model import InvoiceCNN
from config import NUM_CLASSES
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   # .../AI_OCR/src
PROJECT_DIR = BASE_DIR.parent               # .../AI_OCR

CSV_PATH = PROJECT_DIR / "data" / "labels.csv"
IMAGES_DIR = PROJECT_DIR / "data" / "images"

dataset = InvoiceDataset(
    csv_path=str(CSV_PATH),
    images_dir=str(IMAGES_DIR),
)

device = "cpu" #"cuda" if torch.cuda.is_available() else

# podział: 80% train; 20% val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = InvoiceCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

EPOCHS = 10

os.makedirs("output/debug", exist_ok=True)

for epoch in range(EPOCHS):
    # trening
    model.train()
    total_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(train_loader))

    # walidacja
    model.eval()
    correct = 0
    total = 0

    all_true = []
    all_pred = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_true.append(labels.detach().cpu())
            all_pred.append(preds.detach().cpu())

    acc = correct / max(1, total)

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))

    print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, val_acc={acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print(classification_report(
        y_true, y_pred,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    ))

    # zapis metryk epoki
    payload = {
        "epoch": epoch + 1,
        "val_acc": acc,
        "train_loss_avg": avg_loss,
        "confusion_matrix": cm.tolist(),
    }

    json_path = f"output/debug/metrics_epoch_{epoch + 1}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


torch.save(model.state_dict(), "models/invoice_cnn.pth")
print("Model zapisany jako invoice_cnn.pth")
