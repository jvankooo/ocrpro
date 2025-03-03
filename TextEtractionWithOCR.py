import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import pytesseract
import json
import os
from PIL import Image
from collections import Counter
import layoutparser as lp

def extract_metadata(pdf_path):
    """Extract metadata from the PDF file."""
    doc = fitz.open(pdf_path)
    return doc.metadata

def extract_text(pdf_path):
    """Extract text directly from the PDF."""
    doc = fitz.open(pdf_path)
    text_content = "\n\n".join([page.get_text("text") for page in doc])
    return text_content

def extract_tables(pdf_path):
    """Extract tables using pdfplumber with enhanced filtering."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                df = pd.DataFrame(table).replace("", None)
                if df.isnull().sum().sum() > 0.8 * df.size:
                    continue  # Skip tables with too many empty cells
                if df.shape[1] == 1 or df.shape[0] < 2:
                    continue  # Skip noisy single-column/row tables
                if df.map(lambda x: isinstance(x, str) and len(x.strip()) > 50).sum().sum() > 0.9 * df.size:
                    continue  # Skip text blocks misclassified as tables
                tables.append(df.values.tolist())
    return tables

def extract_images_for_ocr(pdf_path):
    """Extract images from PDF and prepare for OCR without using Poppler."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def apply_ocr(pdf_path, ocr_output="ocr_extracted_text.txt"):
    """Perform OCR on PDF pages without using Poppler."""
    images = extract_images_for_ocr(pdf_path)
    ocr_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, config='--psm 6')
        ocr_text += text + "\n\n"
    with open(ocr_output, "w", encoding="utf-8") as f:
        f.write(ocr_text)
    return ocr_output

def detect_layout(pdf_path):
    """Detect document layout using LayoutParser for better text-table segmentation."""
    images = extract_images_for_ocr(pdf_path)
    model = lp.Detectron2LayoutModel("lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config")
    layouts = []
    for image in images:
        layout = model.detect(image)
        layouts.append(layout)
    return layouts

def save_results(metadata, text_content, tables, ocr_output, layouts):
    """Save extracted content to structured formats."""
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text_content)
    with open("extracted_tables.json", "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=4)
    with open("layout_detection.json", "w", encoding="utf-8") as f:
        json.dump([str(layout) for layout in layouts], f, indent=4)
    return Counter({"text_pages": len(text_content.split("\n\n")), "tables": len(tables), "ocr_output": ocr_output, "layouts_detected": len(layouts)})

# Main execution
pdf_path = r"E:\Btech_AI\Intern\ocrpro\Phable CAM Final.pdf"
metadata = extract_metadata(pdf_path)
text_content = extract_text(pdf_path)
tables = extract_tables(pdf_path)
ocr_output = apply_ocr(pdf_path)
layouts = detect_layout(pdf_path)
data_summary = save_results(metadata, text_content, tables, ocr_output, layouts)

print("Extraction Complete:", data_summary)
