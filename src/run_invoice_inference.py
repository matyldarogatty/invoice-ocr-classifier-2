import os
import torch
from PIL import Image
from torchvision import transforms
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from model import InvoiceCNN
from config import LABELS, IMG_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/invoice_cnn.pth"
OUTPUT_DIR = "output/predictions"

# OCR
ocr_model = ocr_predictor(pretrained=True)

#  CNN
model = InvoiceCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


def get_line_bbox_pixels(line, page):
    xs, ys = [], []
    for word in line.words:
        (x1, y1), (x2, y2) = word.geometry
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    H, W = page.dimensions

    x1 = int(min(xs) * W)
    x2 = int(max(xs) * W)
    y1 = int(min(ys) * H)
    y2 = int(max(ys) * H)

    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def analyze_invoice(pdf_path):
    print(f"\nAnaliza faktury: {pdf_path}\n")

    doc = DocumentFile.from_pdf(pdf_path)
    result = ocr_model(doc)
    page = result.pages[0]
    page_image = Image.fromarray(doc[0])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for label in LABELS.values():
        os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

    sample_id = 0

    for block in page.blocks:
        for line in block.lines:
            text = " ".join(w.value for w in line.words)

            x1, y1, x2, y2 = get_line_bbox_pixels(line, page)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = page_image.crop((x1, y1, x2, y2))

            img_tensor = transform(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = model(img_tensor).argmax(1).item()

            label_name = LABELS[pred]

            filename = f"{sample_id:04d}.png"
            save_path = os.path.join(OUTPUT_DIR, label_name, filename)
            crop.save(save_path)

            print(f"[{label_name:15}] {text}")

            sample_id += 1


#  URUCHOMIENIE
if __name__ == "__main__":
    import sys
    analyze_invoice(sys.argv[1])
#python run_invoice_inference.py nowa_faktura.pdf
