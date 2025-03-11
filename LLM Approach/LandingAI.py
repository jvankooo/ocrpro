import requests
import json
import fitz  # PyMuPDF for PDF handling
import os
import time

def extract_text_with_retry(pdf_path, api_key, max_retries=5, retry_delay=10):
    """Retry API calls if rate limit is hit (429 error)."""
    url = "https://landing.ai/agentic-document-extraction/extract"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    with open(pdf_path, "rb") as pdf_file:
        files = {"file": pdf_file}
        
        for attempt in range(max_retries):
            response = requests.post(url, headers=headers, files=files)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"Rate limit hit! Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Error {response.status_code}: {response.text}")
                break
    return None

def preprocess_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF before sending it to API."""
    doc = fitz.open(pdf_path)
    extracted_text = ""
    for page in doc:
        extracted_text += page.get_text("text") + "\n"
    return extracted_text

def save_extracted_data(data, output_path):
    """Save extracted data to JSON file."""
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"Extracted data saved to {output_path}")

if __name__ == "__main__":
    API_KEY = "a2Q0NDNsNWd1am9neXRia2RsZnhmOnpsNG1vMnprNTJualBjRWRqbFV5a1EyZ1pBbFFyYTRj"  
    PDF_PATH = "E:\Btech_AI\Intern\ocrpro\Phable CAM Final.pdf" 
    OUTPUT_JSON = r"E:\Btech_AI\Intern\ocrpro\LLM Approach\results"

    print("Preprocessing PDF...")
    text_preview = preprocess_pdf(PDF_PATH)
    print(f"Extracted preview:\n{text_preview[:500]}...")  # Show first 500 chars
    
    print("Sending PDF to Landing AI API with retry handling...")
    extracted_data = extract_text_with_retry(PDF_PATH, API_KEY)
    
    if extracted_data:
        save_extracted_data(extracted_data, OUTPUT_JSON)
