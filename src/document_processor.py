"""
Document Processing Module
Handles extraction and chunking of PDF, JSON, and TXT files
"""

import PyPDF2
import json
import re
import os
from typing import List
import io


class DocumentProcessor:
    """Handles PDF, JSON, and TXT processing and text chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum words per chunk
            chunk_overlap: Number of overlapping words between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = ['.pdf', '.json', '.txt']
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file content"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_json(self, file_content: bytes) -> str:
        """Extract text from JSON file content"""
        try:
            # Try different encodings
            text_content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    text_content = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_content is None:
                return "Error reading JSON: Unable to decode file with any common encoding"
            
            data = json.loads(text_content)
            
            def extract_recursive(obj, parts=None):
                """Recursively extract text from JSON object"""
                if parts is None:
                    parts = []
                    
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, str):
                            parts.append(f"{key}: {value}")
                        else:
                            extract_recursive(value, parts)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_recursive(item, parts)
                elif isinstance(obj, str):
                    parts.append(obj)
                return parts
            
            return "\n".join(extract_recursive(data))
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {str(e)}"
        except Exception as e:
            return f"Error reading JSON: {str(e)}"
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extract text from TXT file content"""
        try:
            # Try different encodings in order of preference
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    return file_content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, use utf-8 with error handling
            return file_content.decode('utf-8', errors='replace')
        except Exception as e:
            return f"Error reading TXT: {str(e)}"
    
    def extract_text_from_file(self, filename: str, file_content: bytes) -> str:
        """Extract text from file content based on extension"""
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_ext == '.json':
            return self.extract_text_from_json(file_content)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_content)
        else:
            return f"Unsupported file format: {file_ext}. Supported: {', '.join(self.supported_formats)}"
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if len(chunk.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk.strip())
        
        return chunks
    
    def process_document(self, filename: str, file_content: bytes) -> List[str]:
        """Process document and return chunks"""
        if not filename or not file_content:
            raise ValueError("Empty filename or file content")
        
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Validate file extension
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {', '.join(self.supported_formats)}")
        
        # Check file size (basic validation)
        if len(file_content) == 0:
            raise ValueError("File is empty")
        
        if len(file_content) > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError("File too large (max 50MB)")
        
        print(f"📄 Processing {file_ext.upper()} document: {filename} ({len(file_content)} bytes)")
        
        text = self.extract_text_from_file(filename, file_content)
        
        if text.startswith("Error") or text.startswith("Unsupported"):
            raise ValueError(text)
        
        if not text.strip():
            raise ValueError("No text content extracted from file")
        
        cleaned_text = self.clean_text(text)
        chunks = self.chunk_text(cleaned_text)
        
        if not chunks:
            raise ValueError("No valid text chunks created from document")
        
        print(f"✅ Created {len(chunks)} chunks from {file_ext.upper()} file")
        return chunks
    
    def process_document_from_path(self, file_path: str) -> List[str]:
        """Process document from file path (for testing)"""
        with open(file_path, 'rb') as f:
            content = f.read()
        return self.process_document(os.path.basename(file_path), content)
