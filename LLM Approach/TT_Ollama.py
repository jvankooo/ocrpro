import os
import json
import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import ollama
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np


# ✅ Define Input and Output Paths
PDF_PATH = r"E:\Btech_AI\Intern\ocrpro\Phable CAM Final.pdf"
OUTPUT_DIR = r"E:\Btech_AI\Intern\ocrpro\LLM Approach\results\TT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Create subdirectories for outputs
TEXT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "text")
TABLE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "tables")
IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# ✅ Load Table Transformer Model
try:
    model = DetrForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
except ImportError as e:
    print(f"❌ Error loading Table Transformer model: {e}")
    print("Please ensure the `timm` library is installed. Run: `pip install timm`")
    exit(1)

# ✅ Convert PDF Pages to Images
def convert_pdf_to_images(pdf_path):
    """
    Convert PDF pages to images using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_path = os.path.join(IMAGE_OUTPUT_DIR, f"page_{i+1}.png")
        img.save(img_path)
        image_paths.append(img_path)
    return image_paths

# ✅ Extract Text Using Ollama
def extract_text_with_ollama(image_path):
    """
    Extract text from an image using Ollama.
    """
    try:
        with open(image_path, "rb") as img_file:
            response = ollama.generate(
                model="llama2",  # Use a suitable model
                prompt="Extract all text from this image.",
                images=[img_file.read()]
            )
        return response["text"]
    except Exception as e:
        print(f"❌ Error extracting text from {image_path}: {e}")
        return ""

# ✅ Extract Tables Using Table Transformers
def extract_tables(image_path):
    """
    Extract tables from an image using Table Transformers.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Post-process outputs to extract table bounding boxes
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.9:  # Confidence threshold
                tables.append({
                    "box": box.tolist(),
                    "label": model.config.id2label[label.item()]
                })
        return tables
    except Exception as e:
        print(f"❌ Error extracting tables from {image_path}: {e}")
        return []


def extract_images_and_charts(image_path):
    """
    Detect and extract images and charts using YOLOv8.
    """
    try:
        # Load YOLOv8 model (use a pre-trained model or your custom model)
        model = YOLO("yolov8m.pt")  

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Unable to load image {image_path}")
            return []

        # Perform object detection
        results = model(image)

        # Extract detected objects
        detected_objects = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)  # Class ID
                confidence = float(box.conf)  # Confidence score
                if confidence > 0.5:  # Confidence threshold
                    detected_objects.append({
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": box.xyxy.tolist()[0]  # Bounding box coordinates
                    })

        return detected_objects
    except Exception as e:
        print(f"❌ Error extracting images/charts from {image_path}: {e}")
        return []
    
# ✅ Process PDF and Extract Data
def process_pdf(pdf_path):
    """
    Process a PDF and extract text, tables, images, and charts.
    """
    image_paths = convert_pdf_to_images(pdf_path)

    for i, img_path in enumerate(image_paths):
        # Extract text
        text = extract_text_with_ollama(img_path)
        with open(os.path.join(TEXT_OUTPUT_DIR, f"page_{i+1}.txt"), "w") as f:
            f.write(text)

        # Extract tables
        tables = extract_tables(img_path)
        if tables:
            df = pd.DataFrame(tables)
            df.to_excel(os.path.join(TABLE_OUTPUT_DIR, f"page_{i+1}_tables.xlsx"), index=False)

        # Extract images and charts
        images_charts = extract_images_and_charts(img_path)
        if images_charts:
            with open(os.path.join(IMAGE_OUTPUT_DIR, f"page_{i+1}_images_charts.json"), "w") as f:
                json.dump(images_charts, f, indent=4)

    print("✅ Extraction complete. Results saved in respective directories.")
    
# ✅ Run the Pipeline
process_pdf(PDF_PATH)