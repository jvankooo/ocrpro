import os
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import uuid
from PIL import Image
import io
import re
import json

class DocExtractor:
    def __init__(self, model_name="ds4sd/SmolDocling-256M-preview", output_dir="SmallDocling_extracted_data"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name)
        
        # Setup output directories
        self.output_dir = output_dir
        self.text_dir = os.path.join(output_dir, "text")
        self.tables_dir = os.path.join(output_dir, "tables")
        self.images_dir = os.path.join(output_dir, "images")
        self.qna_dir = os.path.join(output_dir, "qna_data")
        
        # Create directories if they don't exist
        for dir_path in [self.text_dir, self.tables_dir, self.images_dir, self.qna_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
    def extract_data(self, pdf_path):
        """Extract all data from PDF and save in organized format"""
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        
        # Open PDF using PyMuPDF (no poppler/pdf2image dependency)
        doc = fitz.open(pdf_path)
        
        # Process each page
        all_text = []
        all_tables = []
        all_images = []
        
        for page_num, page in enumerate(doc):
            print(f"Processing page {page_num+1}/{len(doc)}")
            
            # Get page as PIL Image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Save raw image
            image_filename = f"{pdf_name}_page_{page_num+1}.png"
            img_path = os.path.join(self.images_dir, image_filename)
            img.save(img_path)
            
            # Process with SmolDocling
            inputs = self.processor(images=img, return_tensors="pt")
            
            # Generate outputs
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4
                )
            
            # Decode generated text
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract different elements using SmolDocling's output
            page_text, page_tables, page_images = self._parse_docling_output(generated_text, page_num, pdf_name)
            
            all_text.extend(page_text)
            all_tables.extend(page_tables)
            all_images.extend(page_images)
            
            # Save page-level text
            with open(os.path.join(self.text_dir, f"{pdf_name}_page_{page_num+1}.txt"), "w", encoding="utf-8") as f:
                f.write("\n\n".join([item["text"] for item in page_text]))
        
        # Save complete document text
        with open(os.path.join(self.text_dir, f"{pdf_name}_complete.txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join([item["text"] for item in all_text]))
        
        # Create structured data for QnA bot
        self._create_qna_dataset(all_text, all_tables, all_images, pdf_name)
        
        return {
            "text": all_text,
            "tables": all_tables,
            "images": all_images
        }
    
    def _parse_docling_output(self, output_text, page_num, pdf_name):
        """Parse SmolDocling's output to separate text, tables and image descriptions"""
        page_text = []
        page_tables = []
        page_images = []
        
        # Simple parsing approach - can be enhanced based on actual SmolDocling output format
        # Split by sections that might be indicated in the output
        sections = re.split(r'(TABLE:|FIGURE:|IMAGE:|TEXT:)', output_text)
        
        current_type = "TEXT"  # Default type
        current_content = ""
        
        for i, section in enumerate(sections):
            if section in ["TABLE:", "FIGURE:", "IMAGE:", "TEXT:"]:
                # Save previous content if any
                if current_content.strip():
                    if current_type == "TEXT":
                        page_text.append({
                            "id": str(uuid.uuid4()),
                            "page": page_num + 1,
                            "text": current_content.strip(),
                            "source": f"{pdf_name}_page_{page_num+1}"
                        })
                    elif current_type == "TABLE:":
                        # Try to parse table content into structured format
                        table_id = str(uuid.uuid4())
                        table_data = self._parse_table_content(current_content)
                        
                        # Save as Excel
                        table_filename = f"{pdf_name}_table_{table_id[-6:]}.xlsx"
                        table_path = os.path.join(self.tables_dir, table_filename)
                        
                        try:
                            df = pd.DataFrame(table_data)
                            df.to_excel(table_path, index=False)
                            
                            page_tables.append({
                                "id": table_id,
                                "page": page_num + 1,
                                "filename": table_filename,
                                "data": table_data,
                                "source": f"{pdf_name}_page_{page_num+1}"
                            })
                        except Exception as e:
                            print(f"Error saving table: {e}")
                            # Add as text instead
                            page_text.append({
                                "id": str(uuid.uuid4()),
                                "page": page_num + 1,
                                "text": f"[TABLE CONTENT] {current_content.strip()}",
                                "source": f"{pdf_name}_page_{page_num+1}"
                            })
                            
                    elif current_type in ["FIGURE:", "IMAGE:"]:
                        image_id = str(uuid.uuid4())
                        page_images.append({
                            "id": image_id,
                            "page": page_num + 1,
                            "description": current_content.strip(),
                            "source": f"{pdf_name}_page_{page_num+1}"
                        })
                
                # Update current type and reset content
                current_type = section
                current_content = ""
            else:
                current_content += section
        
        # Process the last section if any
        if current_content.strip():
            if current_type == "TEXT":
                page_text.append({
                    "id": str(uuid.uuid4()),
                    "page": page_num + 1,
                    "text": current_content.strip(),
                    "source": f"{pdf_name}_page_{page_num+1}"
                })
            elif current_type == "TABLE:":
                table_id = str(uuid.uuid4())
                table_data = self._parse_table_content(current_content)
                
                # Save as Excel
                table_filename = f"{pdf_name}_table_{table_id[-6:]}.xlsx"
                table_path = os.path.join(self.tables_dir, table_filename)
                
                try:
                    df = pd.DataFrame(table_data)
                    df.to_excel(table_path, index=False)
                    
                    page_tables.append({
                        "id": table_id,
                        "page": page_num + 1,
                        "filename": table_filename,
                        "data": table_data,
                        "source": f"{pdf_name}_page_{page_num+1}"
                    })
                except Exception as e:
                    print(f"Error saving table: {e}")
                    # Add as text instead
                    page_text.append({
                        "id": str(uuid.uuid4()),
                        "page": page_num + 1,
                        "text": f"[TABLE CONTENT] {current_content.strip()}",
                        "source": f"{pdf_name}_page_{page_num+1}"
                    })
            elif current_type in ["FIGURE:", "IMAGE:"]:
                image_id = str(uuid.uuid4())
                page_images.append({
                    "id": image_id,
                    "page": page_num + 1,
                    "description": current_content.strip(),
                    "source": f"{pdf_name}_page_{page_num+1}"
                })
        
        return page_text, page_tables, page_images
    
    def _parse_table_content(self, table_content):
        """Convert table text content to structured data for Excel"""
        lines = [line.strip() for line in table_content.strip().split('\n') if line.strip()]
        
        if not lines:
            return [{"Empty": "Table"}]
        
        # Try to determine delimiter (tab, pipe, etc.)
        if '|' in lines[0]:
            delimiter = '|'
        elif '\t' in lines[0]:
            delimiter = '\t'
        else:
            # Fallback to space with heuristic approach
            delimiter = None
            
        # Parse header
        if delimiter:
            headers = [cell.strip() for cell in lines[0].split(delimiter) if cell.strip()]
        else:
            # Try to infer columns by splitting on multiple spaces
            headers = re.split(r'\s{2,}', lines[0].strip())
            
        # Replace empty headers
        headers = [f"Column_{i}" if not h else h for i, h in enumerate(headers)]
        
        # Parse rows
        rows = []
        for line in lines[1:]:
            if delimiter:
                cells = [cell.strip() for cell in line.split(delimiter)]
            else:
                cells = re.split(r'\s{2,}', line.strip())
                
            # Ensure all rows have same number of columns as header
            while len(cells) < len(headers):
                cells.append("")
                
            # Create row as dict
            row = {headers[i]: cells[i] for i in range(min(len(headers), len(cells)))}
            rows.append(row)
            
        return rows
    
    def _create_qna_dataset(self, all_text, all_tables, all_images, pdf_name):
        """Create a structured dataset suitable for QnA bot training"""
        # Combine all data into a structured format
        structured_data = {
            "document_name": pdf_name,
            "sections": [],
            "tables": [],
            "images": []
        }
        
        # Add text sections
        for text_item in all_text:
            structured_data["sections"].append({
                "id": text_item["id"],
                "page": text_item["page"],
                "content": text_item["text"]
            })
            
        # Add tables
        for table_item in all_tables:
            structured_data["tables"].append({
                "id": table_item["id"],
                "page": table_item["page"],
                "filename": table_item["filename"],
                "content": str(table_item["data"])  # Simplified for JSON storage
            })
            
        # Add images
        for image_item in all_images:
            structured_data["images"].append({
                "id": image_item["id"],
                "page": image_item["page"],
                "description": image_item["description"]
            })
            
        # Save structured data for QnA bot
        with open(os.path.join(self.qna_dir, f"{pdf_name}_qna_data.json"), "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2)
            
        # Also create a simplified text-only version for basic QnA
        simple_contexts = []
        
        # Add text
        for text_item in all_text:
            simple_contexts.append({
                "id": text_item["id"],
                "content": text_item["text"],
                "type": "text",
                "page": text_item["page"]
            })
            
        # Add table descriptions
        for table_item in all_tables:
            simple_contexts.append({
                "id": table_item["id"],
                "content": f"Table data from page {table_item['page']}: {str(table_item['data'])}",
                "type": "table",
                "page": table_item["page"]
            })
            
        # Add image descriptions
        for image_item in all_images:
            simple_contexts.append({
                "id": image_item["id"],
                "content": f"Image on page {image_item['page']}: {image_item['description']}",
                "type": "image",
                "page": image_item["page"]
            })
            
        # Save simplified data
        with open(os.path.join(self.qna_dir, f"{pdf_name}_simple_contexts.json"), "w", encoding="utf-8") as f:
            json.dump(simple_contexts, f, indent=2)

# Example usage
def main():
    extractor = DocExtractor()
    pdf_path = r"E:\Btech_AI\Intern\ocrpro\Phable CAM Final.pdf"  # Replace with your PDF path
    extracted_data = extractor.extract_data(pdf_path)
    
    print(f"Extraction complete. Data saved to {extractor.output_dir}")
    print(f"Text files: {len(extracted_data['text'])}")
    print(f"Tables: {len(extracted_data['tables'])}")
    print(f"Images: {len(extracted_data['images'])}")
    
if __name__ == "__main__":
    main()