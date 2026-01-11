from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
from PIL import Image
import csv

model = ocr_predictor(pretrained=True)

input_dir = "data/raw"
output_img_dir = "data/images"
os.makedirs(output_img_dir, exist_ok=True)

csv_path = "data/labels.csv"

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

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    return x1, y1, x2, y2


rows = []
global_id = 0

for pdf_name in os.listdir(input_dir):
    if not pdf_name.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_dir, pdf_name)
    print(f"Przetwarzam: {pdf_name}")

    doc = DocumentFile.from_pdf(pdf_path)
    result = model(doc)

    page = result.pages[0]
    page_image = Image.fromarray(doc[0])

    for block in page.blocks:
        for line in block.lines:
            text = " ".join(w.value for w in line.words)

            bbox = get_line_bbox_pixels(line, page)
            x1, y1, x2, y2 = bbox

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = page_image.crop((x1, y1, x2, y2))

            filename = f"{pdf_name[:-4]}_{global_id:05d}.png"
            crop.save(os.path.join(output_img_dir, filename))

            rows.append([filename, text, ""])
            global_id += 1


# zapis CSV
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "text", "label"])
    writer.writerows(rows)
