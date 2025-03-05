import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import json
import os
from collections import Counter

def extract_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    return doc.metadata

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text_content = "\n\n".join([page.get_text("text") for page in doc])
    return text_content

def extract_tables(pdf_path):
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
                if df.applymap(lambda x: isinstance(x, str) and len(x.strip()) > 50).sum().sum() > 0.9 * df.size:
                    continue  # Skip text blocks misclassified as tables
                tables.append(df.values.tolist())
    return tables

def extract_images(pdf_path, image_dir="extracted_images"):
    os.makedirs(image_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_count = 0
    
    for i, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            image_ext = base_image["ext"]
            img_filename = f"{image_dir}/image_{i+1}_{img_index+1}.{image_ext}"
            with open(img_filename, "wb") as img_file:
                img_file.write(image_data)
            image_count += 1
    return image_count

def save_results(metadata, text_content, tables, image_count):
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text_content)
    with open("extracted_tables.json", "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=4)
    return Counter({"text_pages": len(text_content.split("\n\n")), "tables": len(tables), "images": image_count})

# Main execution
pdf_path = r"E:\Btech_AI\Intern\ocrpro\Phable CAM Final.pdf"
metadata = extract_metadata(pdf_path)
text_content = extract_text(pdf_path)
tables = extract_tables(pdf_path)
image_count = extract_images(pdf_path)
data_summary = save_results(metadata, text_content, tables, image_count)

print("Extraction Complete:", data_summary)
