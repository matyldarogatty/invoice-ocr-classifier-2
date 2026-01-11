import torch
from torch.utils.data import DataLoader, random_split
from dataset import InvoiceDataset
from model import InvoiceCNN
from config import NUM_CLASSES

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = InvoiceDataset(
    csv_path="data/labels.csv",
    images_dir="data/images"
)

# podział: 80% train / 20% val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = InvoiceCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # walidacja
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}: loss={total_loss:.3f}, val_acc={acc:.3f}")

torch.save(model.state_dict(), "invoice_cnn.pth")
print("Model zapisany jako invoice_cnn.pth")
