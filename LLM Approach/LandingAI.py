import requests
import json
import fitz  # PyMuPDF for PDF handling
import os
import time
import base64
import pandas as pd

# ✅ Configuration
API_KEY = ".."
LANDING_AI_URL = "https://api.va.landing.ai/v1/tools/agentic-document-analysis"
PDF_PATH = r"E:\Btech_AI\Intern\ocrpro\Phable CAM Final.pdf"
OUTPUT_DIR = r"E:\Btech_AI\Intern\ocrpro\LLM Approach\results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "extracted_data.json")
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "extracted_tables.xlsx")
OUTPUT_IMAGES = os.path.join(OUTPUT_DIR, "extracted_images.json")

# ✅ Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Headers for API Call
HEADERS = {
  "Authorization": "Basic {{your_api_key}}",
}

def extract_data_with_retry(pdf_path, max_retries=5, retry_delay=10):
    """Retry API calls if rate limit is hit (429 error)."""
    with open(pdf_path, "rb") as pdf_file:
        files = {"file": pdf_file}
        
        for attempt in range(max_retries):
            response = requests.post(LANDING_AI_URL, headers=HEADERS, files=files)

            print(f"🔄 Attempt {attempt+1}: Status Code {response.status_code}")

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"⚠️ Rate limit hit! Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            elif response.status_code in [401, 403]:
                print(f"❌ Error {response.status_code}: Check API Key and Permissions.")
                break
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break
    return None

def preprocess_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF before sending it to API."""
    doc = fitz.open(pdf_path)
    extracted_text = ""
    for page in doc:
        extracted_text += page.get_text("text") + "\n"
    return extracted_text

def save_extracted_data(data, output_json, output_excel, output_images):
    """Save extracted data (Text, Tables, Images)."""
    
    extracted_text = data.get("text", "")
    tables = data.get("tables", [])
    images = data.get("images", [])

    # ✅ Save Text
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"📄 Extracted data saved: {output_json}")

    # ✅ Save Tables in Excel
    if tables:
        with pd.ExcelWriter(output_excel) as writer:
            for idx, table in enumerate(tables):
                df = pd.DataFrame(table)
                df.to_excel(writer, sheet_name=f"Table_{idx+1}", index=False)
        print(f"📊 Extracted tables saved: {output_excel}")

    # ✅ Save Image Descriptions
    if images:
        with open(output_images, "w", encoding="utf-8") as img_file:
            json.dump(images, img_file, indent=4, ensure_ascii=False)
        print(f"🖼️ Extracted images saved: {output_images}")

if __name__ == "__main__":
    print("📄 Preprocessing PDF...")
    text_preview = preprocess_pdf(PDF_PATH)
    print(f"📝 Extracted preview:\n{text_preview[:1000]}...\n")  # Show first 1000 chars
    
    print("📡 Sending PDF to Landing AI API with retry handling...")
    extracted_data = extract_data_with_retry(PDF_PATH)
    
    if extracted_data:
        save_extracted_data(extracted_data, OUTPUT_JSON, OUTPUT_EXCEL, OUTPUT_IMAGES)
        print("✅ Extraction Completed Successfully!")
    else:
        print("❌ Extraction Failed.")
