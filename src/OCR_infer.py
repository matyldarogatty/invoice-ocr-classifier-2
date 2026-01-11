from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import os

model = ocr_predictor(pretrained=True)

doc = DocumentFile.from_pdf("nowa_faktura.pdf")
result = model(doc)
page = result.pages[0]

page_image = Image.fromarray(doc[0])

output_dir = "inference_parts"
os.makedirs(output_dir, exist_ok=True)

sample_id = 0

def get_bbox(line, page):
    xs, ys = [], []
    for word in line.words:
        (x1, y1), (x2, y2) = word.geometry
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    H, W = page.dimensions
    return (
        int(min(xs) * W),
        int(min(ys) * H),
        int(max(xs) * W),
        int(max(ys) * H)
    )

for block in page.blocks:
    for line in block.lines:
        bbox = get_bbox(line, page)
        crop = page_image.crop(bbox)
        crop.save(f"inference_parts/{sample_id:04d}.png")
        sample_id += 1
