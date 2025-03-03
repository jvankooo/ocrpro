import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import pytesseract
import json
import os
from PIL import Image
from collections import Counter
import layoutparser as lp

# Define paths to local model files (Detectron 2)
CONFIG_PATH = r"E:\Btech_AI\Intern\ocrpro\faster_rcnn_R_50_FPN_3x.yaml"
MODEL_PATH = r"E:\Btech_AI\Intern\ocrpro\model_final_280758.pkl"

def extract_metadata(pdf_path):
    """Extract metadata from the PDF file."""
    doc = fitz.open(pdf_path)
    return doc.metadata

def extract_text(pdf_path):
    """Extract text directly from the PDF."""
    doc = fitz.open(pdf_path)
    text_content = "\n\n".join([page.get_text("text") for page in doc])
    return text_content, len(text_content.split())

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
                tables.append(df)
    return tables, len(tables)

def extract_images_for_ocr(pdf_path):
    """Extract images from PDF and prepare for OCR."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images, len(images)

def apply_ocr(pdf_path, ocr_output="ocr_extracted_text.txt"):
    """Perform OCR on PDF pages."""
    images, image_count = extract_images_for_ocr(pdf_path)
    ocr_text = ""
    for image in images:
        text = pytesseract.image_to_string(image, config='--psm 6')
        ocr_text += text + "\n\n"
    
    with open(ocr_output, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    return ocr_output, ocr_text, len(ocr_text.split()), image_count

def detect_layout(pdf_path):
    """Detect document layout using LayoutParser and a local Detectron2 model."""
    images, _ = extract_images_for_ocr(pdf_path)

    try:
        # Load local Detectron2 model
        model = lp.Detectron2LayoutModel(
            config_path=CONFIG_PATH,
            model_path=MODEL_PATH,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]  # Adjust confidence threshold
        )
    except Exception as e:
        print(f"Error loading Detectron2 model: {e}")
        return []

    layouts = []
    for image in images:
        layout = model.detect(image)
        formatted_layout = [
            {
                "type": block.type,
                "score": block.score,
                "coordinates": block.coordinates
            }
            for block in layout
        ]
        layouts.append(formatted_layout)
    
    return layouts, len(layouts)

def save_results(metadata, text_content, tables, ocr_output, layouts):
    """Save extracted content to structured formats."""
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text_content)
    
    # Save tables in Excel format
    if tables:
        with pd.ExcelWriter("extracted_tables.xlsx") as writer:
            for i, df in enumerate(tables):
                df.to_excel(writer, sheet_name=f"Table_{i+1}", index=False)

    with open("layout_detection.json", "w", encoding="utf-8") as f:
        json.dump(layouts, f, indent=4)
    
    return Counter({
        "text_pages": len(text_content.split("\n\n")),
        "tables": len(tables),
        "ocr_output": ocr_output,
        "layouts_detected": len(layouts)
    })

# Main execution
pdf_path = r"E:\Btech_AI\Intern\ocrpro\Phable CAM Final.pdf"

metadata = extract_metadata(pdf_path)
text_content, word_count_before_ocr = extract_text(pdf_path)
tables, table_count_before_ocr = extract_tables(pdf_path)
ocr_output, ocr_text, word_count_after_ocr, image_count_after_ocr = apply_ocr(pdf_path)
layouts, layout_count = detect_layout(pdf_path)

# Summary
summary = {
    "Total Words Extracted (Before OCR)": word_count_before_ocr,
    "Total Words Extracted (After OCR)": word_count_after_ocr,
    "Total Images (Before OCR)": image_count_after_ocr,
    "Total Images (After OCR)": image_count_after_ocr,  # No image processing is done, so same count
    "Total Tables (Before OCR)": table_count_before_ocr,
    "Total Tables (After OCR)": table_count_before_ocr,  # OCR doesn't detect tables separately
    "Layout Detections": layout_count
}

data_summary = save_results(metadata, text_content, tables, ocr_output, layouts)

# Print Summary
print("\n✅ Extraction Complete! Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")
