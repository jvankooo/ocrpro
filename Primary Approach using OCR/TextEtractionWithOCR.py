import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import pytesseract
import json
import os
import difflib
from PIL import Image
from collections import Counter
import layoutparser as lp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define paths to local model files
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
                tables.append(df)
    return tables

def extract_images_for_ocr(pdf_path):
    """Extract images from PDF and prepare for OCR."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def apply_ocr(pdf_path, ocr_output="ocr_extracted_text.txt"):
    """Perform OCR on PDF pages."""
    images = extract_images_for_ocr(pdf_path)
    ocr_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, config='--psm 6')
        ocr_text += text + "\n\n"
    with open(ocr_output, "w", encoding="utf-8") as f:
        f.write(ocr_text)
    return ocr_text

def detect_layout(pdf_path):
    """Detect document layout using LayoutParser and a local Detectron2 model."""
    images = extract_images_for_ocr(pdf_path)

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
        layouts.append(layout)
    
    return layouts

def save_results(metadata, text_content, tables, ocr_text, layouts):
    """Save extracted content to structured formats."""
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text_content)
    with open("ocr_extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(ocr_text)
    with open("extracted_tables.json", "w", encoding="utf-8") as f:
        json.dump([df.to_dict() for df in tables], f, indent=4)
    with open("layout_detection.json", "w", encoding="utf-8") as f:
        json.dump([str(layout) for layout in layouts], f, indent=4)
    
    # Save tables to Excel
    if tables:
        with pd.ExcelWriter("extracted_tables.xlsx") as writer:
            for idx, df in enumerate(tables):
                df.to_excel(writer, sheet_name=f"Table_{idx+1}", index=False)

    return Counter({
        "text_words_before_ocr": len(text_content.split()),
        "text_words_after_ocr": len(ocr_text.split()),
        "word_difference": len(ocr_text.split()) - len(text_content.split()),
        "tables_detected": len(tables),
        "layouts_detected": len(layouts)
    })

def compare_text_similarity(text1, text2):
    """Compare similarity between extracted text and OCR text using cosine similarity."""
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)[0, 1]
    return round(similarity * 100, 2)  # Convert to percentage

# Main execution
pdf_path = r"E:\Btech_AI\Intern\ocrpro\Phable CAM Final.pdf"

metadata = extract_metadata(pdf_path)
text_content = extract_text(pdf_path)
tables = extract_tables(pdf_path)
ocr_text = apply_ocr(pdf_path)
layouts = detect_layout(pdf_path)
data_summary = save_results(metadata, text_content, tables, ocr_text, layouts)

# Compare text extracted before and after OCR
text_similarity = compare_text_similarity(text_content, ocr_text)

# Print summary
print("✅ Extraction Complete:", data_summary)
print(f"📊 Text Similarity (Before & After OCR): {text_similarity}%")
