#!/usr/bin/env python3
"""
🤖 RAG-Based Document QA Chatbot
A beautiful and functional document question-answering system using RAG architecture.
"""

import gradio as gr
import PyPDF2
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
import os
import json
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class DocumentProcessor:
    """Handles PDF, JSON, and TXT processing and text chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = ['.pdf', '.json', '.txt']
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_json(self, json_path: str) -> str:
        """Extract text from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            def extract_text_recursive(obj, text_parts=[]):
                """Recursively extract text from JSON object"""
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, str):
                            text_parts.append(f"{key}: {value}")
                        else:
                            extract_text_recursive(value, text_parts)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_text_recursive(item, text_parts)
                elif isinstance(obj, str):
                    text_parts.append(obj)
                return text_parts
            
            text_parts = extract_text_recursive(data)
            return "\n".join(text_parts)
            
        except Exception as e:
            return f"Error reading JSON: {str(e)}"
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(txt_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                return f"Error reading TXT file: {str(e)}"
        except Exception as e:
            return f"Error reading TXT file: {str(e)}"
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from file based on extension"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.json':
            return self.extract_text_from_json(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            return f"Unsupported file format: {file_ext}. Supported formats: {', '.join(self.supported_formats)}"
    
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
    
    def process_document(self, file_path: str) -> List[str]:
        """Process document and return chunks"""
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"📄 Processing {file_ext.upper()} document: {file_path}")
        
        text = self.extract_text_from_file(file_path)
        
        # Check if extraction was successful
        if text.startswith("Error") or text.startswith("Unsupported"):
            raise ValueError(text)
        
        cleaned_text = self.clean_text(text)
        chunks = self.chunk_text(cleaned_text)
        print(f"✅ Created {len(chunks)} chunks from {file_ext.upper()} file")
        return chunks


class RAGSystem:
    """Complete RAG system for document QA"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_model = None
        self.vector_store = None
        self.chunks = []
        self.setup_models()
    
    def setup_models(self):
        """Initialize embedding model"""
        print("🚀 Initializing models...")
        
        # Load embedding model (lightweight and fast)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        print("✅ Using context-based QA approach")
    
    def load_document(self, pdf_path: str):
        """Load and process document"""
        self.chunks = self.document_processor.process_document(pdf_path)
        
        if not self.chunks:
            raise ValueError("No valid chunks extracted from document")
        
        # Create embeddings
        print("🔍 Creating embeddings...")
        embeddings = self.embedding_model.encode(self.chunks)
        
        # Setup FAISS vector store
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.vector_store.add(embeddings.astype('float32'))
        
        print(f"✅ Vector store created with {len(self.chunks)} documents")
    
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve most relevant chunks for the query"""
        if not self.vector_store:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        scores, indices = self.vector_store.search(query_embedding.astype('float32'), k)
        
        # Return chunks with scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate answer using retrieved context"""
        # Combine context
        context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks
        
        # Simple extractive approach - find most relevant sentences
        sentences = context.split('.')
        query_words = set(query.lower().split())
        
        # Score sentences based on query word overlap
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Only consider substantial sentences
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                if overlap > 0:
                    scored_sentences.append((sentence.strip(), overlap))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if scored_sentences:
            # Combine top 2-3 most relevant sentences
            answer_parts = [sent[0] for sent in scored_sentences[:3]]
            answer = ". ".join(answer_parts)
            if not answer.endswith('.'):
                answer += "."
            return answer
        else:
            # Fallback: return first part of context
            return context[:500] + "..." if len(context) > 500 else context
    
    def answer_question(self, query: str) -> Dict[str, any]:
        """Complete QA pipeline"""
        if not self.chunks:
            return {
                "answer": "❌ No document loaded. Please upload a PDF first.",
                "sources": [],
                "confidence": 0.0
            }
        
        relevant_chunks = self.retrieve_relevant_chunks(query, k=5)
        
        if not relevant_chunks:
            return {
                "answer": "❌ No relevant information found in the document.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Extract chunks and calculate average confidence
        context_chunks = [chunk for chunk, _ in relevant_chunks]
        # Retrieve relevant chunksant_chunks]
        avg_confidence = np.mean([score for _, score in relevant_chunks])
        
        # Generate answer
        answer = self.generate_answer(query, context_chunks)
        
        return {
            "answer": answer,
            "sources": context_chunks[:3],  # Top 3 sources
            "confidence": float(avg_confidence)
        }


# Initialize RAG system
rag_system = RAGSystem()

def upload_and_process_document(file):
    """Handle document upload and processing"""
    if file is None:
        return "❌ Please upload a file (PDF, JSON, or TXT)."
    
    # Check file extension
    file_ext = os.path.splitext(file.name)[1].lower()
    supported_formats = ['.pdf', '.json', '.txt']
    
    if file_ext not in supported_formats:
        return f"❌ Unsupported file format: {file_ext}. Please upload: {', '.join(supported_formats)}"
    
    try:
        # Process the uploaded file
        rag_system.load_document(file.name)
        return f"✅ {file_ext.upper()} document processed successfully! Ready to answer questions about: {os.path.basename(file.name)}"
    except Exception as e:
        return f"❌ Error processing document: {str(e)}"

def answer_question(question, chat_history):
    """Handle question answering"""
    if not question.strip():
        return chat_history, ""
    
    # Get answer from RAG system
    result = rag_system.answer_question(question)
    
    # Format response
    answer = result["answer"]
    confidence = result["confidence"]
    
    # Add confidence indicator
    if confidence > 0.8:
        confidence_emoji = "🟢"
    elif confidence > 0.6:
        confidence_emoji = "🟡"
    else:
        confidence_emoji = "🔴"
    
    formatted_answer = f"{answer}\n\n{confidence_emoji} Confidence: {confidence:.2f}"
    
    # Update chat history
    chat_history.append([question, formatted_answer])
    
    return chat_history, ""

def show_sources(question):
    """Show source chunks for transparency"""
    if not question.strip():
        return "Please ask a question first."
    
    result = rag_system.answer_question(question)
    sources = result["sources"]
    
    if not sources:
        return "No sources found."
    
    formatted_sources = "\n\n" + "="*50 + "\n\n".join(
        [f"📄 **Source {i+1}:**\n{source[:300]}..." for i, source in enumerate(sources)]
    )
    
    return formatted_sources


# Custom CSS for beautiful styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.header {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.chat-container {
    border: 2px solid #e1e5e9;
    border-radius: 10px;
    padding: 10px;
}

.upload-area {
    border: 2px dashed #667eea;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    background-color: #f8f9fa;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="🤖 RAG Document QA Chatbot") as demo:
    
    # Header
    gr.HTML("""
    <div class="header">
        <h1>🤖 RAG-Based Document QA Chatbot</h1>
        <p>Upload your documents (PDF, JSON, TXT) and ask questions! Powered by AI and semantic search.</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3>📄 Document Upload</h3>")
            
            file_upload = gr.File(
                label="Upload Document (PDF, JSON, TXT)",
                file_types=[".pdf", ".json", ".txt"],
                elem_classes=["upload-area"]
            )
            
            upload_status = gr.Textbox(
                label="📊 Status",
                interactive=False,
                value="🔄 Ready to upload document..."
            )
            
            process_btn = gr.Button(
                "🚀 Process Document",
                variant="primary",
                size="lg"
            )
            
        with gr.Column(scale=2):
            gr.HTML("<h3>💬 Chat with Your Document</h3>")
            
            chatbot = gr.Chatbot(
                label="🤖 AI Assistant",
                height=400,
                elem_classes=["chat-container"]
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="💭 Ask a question",
                    placeholder="What would you like to know about your document?",
                    scale=4
                )
                ask_btn = gr.Button("🔍 Ask", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                sources_btn = gr.Button("📚 Show Sources", variant="secondary")
    
    # Sources section
    with gr.Row():
        sources_output = gr.Textbox(
            label="📚 Source Documents",
            lines=8,
            interactive=False,
            visible=False
        )
    
    # Event handlers
    process_btn.click(
        fn=upload_and_process_document,
        inputs=[file_upload],
        outputs=[upload_status]
    )
    
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input]
    )
    
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input]
    )
    
    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, sources_output]
    )
    
    def toggle_sources(question):
        sources = show_sources(question)
        return gr.update(value=sources, visible=True)
    
    sources_btn.click(
        fn=toggle_sources,
        inputs=[question_input],
        outputs=[sources_output]
    )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 20px; color: #666;">
        <p>🔧 Built with ❤️ using Gradio, Transformers, and FAISS</p>
        <p>🚀 Supports PDF, JSON, TXT files • 🧠 AI-powered answers • 🔍 Semantic search</p>
    </div>
    """)


if __name__ == "__main__":
    print("🚀 Launching RAG Document QA Chatbot...")
    print("📱 The app will open in your browser automatically!")
    print("🔗 You can also share the public link with others!")
    
    demo.launch(
        share=True,  # Create a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        show_error=True,  # Show errors in UI
        debug=False  # Disable debug for production
    )
