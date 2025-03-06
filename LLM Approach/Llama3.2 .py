import requests
import fitz  # PyMuPDF
import os
import json
import pandas as pd
from PIL import Image

# LLaMA 3.2 Vision API Endpoint and Key (Replace with actual credentials)
LLAMA_API_URL = "https://api.llama32vision.com/v1/extract"
API_KEY = "your_api_key_here"

# Input PDF Path
PDF_PATH = r"E:\Btech_AI\Intern\ocrpro\Phable CAM Final.pdf"
OUTPUT_DIR = "llama_extracted_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Convert PDF Pages to Images and Extract Embedded Images
def convert_pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    image_paths = []
    extracted_images = []
    
    for i, page in enumerate(doc):
        # Convert entire page to an image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_path = os.path.join(OUTPUT_DIR, f"page_{i+1}.png")
        img.save(img_path)
        image_paths.append(img_path)
        
        # Extract embedded images separately
        for img_index, img_data in enumerate(page.get_images(full=True)):
            xref = img_data[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]
            extracted_img_path = os.path.join(OUTPUT_DIR, f"embedded_img_{i+1}_{img_index+1}.{img_ext}")
            with open(extracted_img_path, "wb") as img_file:
                img_file.write(img_bytes)
            extracted_images.append(extracted_img_path)
    
    return image_paths, extracted_images

# Step 2: Send Images to LLaMA 3.2 Vision API for Text, Tables, and Figures
def send_to_llama(image_paths, extracted_images):
    extracted_data = []
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    for img_path in image_paths + extracted_images:
        with open(img_path, "rb") as img_file:
            files = {"file": img_file}
            payload = {"prompt": "Extract text, tables as structured JSON, and describe images."}
            response = requests.post(LLAMA_API_URL, headers=headers, files=files, data=payload)
            
            if response.status_code == 200:
                result = response.json()
                extracted_data.append(result)
            else:
                print(f"Error processing {img_path}: {response.text}")
    
    return extracted_data

# Step 3: Process Extracted Data (Save Text and Convert Tables)
def process_extracted_data(data, output_text_file, output_table_file):
    extracted_text = ""
    extracted_tables = []
    extracted_images = []
    
    for entry in data:
        extracted_text += entry.get("text", "") + "\n\n"
        tables = entry.get("tables", [])
        images = entry.get("images", [])
        
        for table in tables:
            df = pd.DataFrame(table)
            extracted_tables.append(df)
        
        extracted_images.extend(images)
    
    # Save extracted text
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    # Save tables in an Excel file
    if extracted_tables:
        with pd.ExcelWriter(output_table_file) as writer:
            for idx, df in enumerate(extracted_tables):
                df.to_excel(writer, sheet_name=f"Table_{idx+1}", index=False)
    
    # Save extracted image descriptions
    with open("extracted_image_descriptions.json", "w", encoding="utf-8") as f:
        json.dump(extracted_images, f, indent=4)
    
    return extracted_text, extracted_tables, extracted_images

# Execute the Pipeline
image_files, embedded_images = convert_pdf_to_images(PDF_PATH)
llama_extracted_data = send_to_llama(image_files, embedded_images)
text, tables, images = process_extracted_data(llama_extracted_data, os.path.join(OUTPUT_DIR, "llama_extracted_text.txt"), os.path.join(OUTPUT_DIR, "llama_extracted_tables.xlsx"))

print("✅ LLaMA 3.2 Vision API Processing Complete. Extracted text, tables, and images saved.")
