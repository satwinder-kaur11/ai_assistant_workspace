"""
app/ingestion/loaders.py

Parses raw text out of uploaded files (PDFs, DOCX, TXT).
"""

import os
import PyPDF2
import docx

def parse_document(file_path: str) -> str:
    """Parses a document (PDF, DOCX, TXT) and returns its text content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
            
    elif ext == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
        
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
        
    else:
        raise ValueError(f"Unsupported file format: {ext}")
