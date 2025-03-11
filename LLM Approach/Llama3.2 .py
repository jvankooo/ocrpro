import requests
import fitz  # PyMuPDF
import os
import json
import time
import pandas as pd
from PIL import Image

# ✅ Replace with your actual API key
API_KEY = "b1b1e69d-dfb9-439f-9991-bee468c10533"
LLAMA_API_URL = "https://api.llama-api.com"

# ✅ Define Input and Output Paths
PDF_PATH = r"E:\Btech_AI\Intern\ocrpro\Phable CAM Final.pdf"
OUTPUT_DIR = r"E:\Btech_AI\Intern\ocrpro\LLM Approach\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Ensure API Key is Sent Correctly
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ✅ Convert PDF Pages to Images
def convert_pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_path = os.path.join(OUTPUT_DIR, f"page_{i+1}.png")
        img.save(img_path)
        image_paths.append(img_path)
    
    return image_paths

# ✅ Send Image to LLaMA API with Rate Limit Handling
def send_to_llama(image_path, retries=3):
    for attempt in range(retries):
        with open(image_path, "rb") as img_file:
            files = {"file": img_file}
            payload = {"prompt": "Extract text, tables as structured JSON, and describe images."}
            
            response = requests.post(LLAMA_API_URL, headers=HEADERS, files=files, data=payload)

            print(f"🔄 Processing: {image_path} (Attempt {attempt+1}/{retries})")
            print(f"📡 Status Code: {response.status_code}")

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("⚠️ Rate limit exceeded. Retrying in a few seconds...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"❌ Error processing {image_path}: {response.text}")
                return None  # Stop retrying for other errors

    return None  # Return None if all retries fail

# ✅ Process Extracted Data and Save Results
def process_extracted_data(data, text_file, table_file):
    extracted_text = ""
    extracted_tables = []
    
    for entry in data:
        extracted_text += entry.get("text", "") + "\n\n"
        tables = entry.get("tables", [])
        
        for table in tables:
            df = pd.DataFrame(table)
            extracted_tables.append(df)
    
    # Save text
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    # Save tables in an Excel file
    if extracted_tables:
        with pd.ExcelWriter(table_file) as writer:
            for idx, df in enumerate(extracted_tables):
                df.to_excel(writer, sheet_name=f"Table_{idx+1}", index=False)
    
    return extracted_text, extracted_tables

# ✅ Run the Full Pipeline
image_files = convert_pdf_to_images(PDF_PATH)
extracted_data = [send_to_llama(img) for img in image_files if send_to_llama(img)]
text, tables = process_extracted_data(extracted_data, os.path.join(OUTPUT_DIR, "llama_extracted_text.txt"), os.path.join(OUTPUT_DIR, "llama_extracted_tables.xlsx"))

print("✅ Extraction Complete. Text and tables saved.")
