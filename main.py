#!/usr/bin/env python3
"""
RAG Document QA Chatbot - Main Application
Clean, organized, and modular implementation
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_system import RAGSystem
from web_interface import WebInterface


def load_sample_document(rag_system: RAGSystem) -> bool:
    """Try to load all available sample documents for testing"""
    data_dir = Path("data")
    sample_files = ['problem.pdf', 'sample_data.json', 'sample_text.txt']
    loaded_any = False
    
    for filename in sample_files:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                print(f"📄 Found sample file: {filename}")
                rag_system.load_document_from_path(str(filepath))
                loaded_any = True
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                continue
    
    return loaded_any


def main():
    """Main application entry point"""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="RAG Document QA Chatbot")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--basic-mode", action="store_true", help="Force basic mode (no ML models)")
    
    args = parser.parse_args()
    
    print("🤖 RAG Based QA Chatbot")
    print("=" * 50)
    print("🏗️  Clean, organized, and modular implementation")
    print()

    # Get Gemini API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # Initialize RAG system
    try:
        use_advanced = not args.basic_mode
        rag_system = RAGSystem(use_advanced=use_advanced, google_api_key=google_api_key)
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return 1
    
    # Try to load a sample document
    has_sample = load_sample_document(rag_system)
    if has_sample:
        print("💡 You can immediately start asking questions about the pre-loaded document!")
    
    print()
    print("🎯 Instructions:")
    print(f"1. Open http://localhost:{args.port} in your browser")
    print("2. Upload a PDF, JSON, or TXT file")
    print("3. Ask questions about your document")
    print("4. Press Ctrl+C to stop the server")
    print()
    
    # Start web interface
    try:
        web_interface = WebInterface(rag_system, port=args.port)
        web_interface.start_server(open_browser=not args.no_browser)
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
