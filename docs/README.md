# 🤖 RAG Document QA Chatbot

A clean, modular, and professional implementation of a **Retrieval-Augmented Generation (RAG)** system for document-based question answering.

## ✨ Features

- **📄 Multi-Format Support**: PDF, JSON, and TXT files
- **🧠 Dual Mode Operation**: Advanced (ML-based) and Basic (keyword-based)
- **🎨 Modern Web Interface**: Beautiful, responsive design
- **🔍 Intelligent Search**: Semantic similarity or keyword matching
- **📊 Confidence Scoring**: Know how reliable the answers are
- **🏗️ Modular Architecture**: Clean, maintainable code structure
- **⚡ Fast Performance**: Optimized for speed and efficiency

## 🚀 Quick Start

### Option 1: Basic Setup (Minimum Dependencies)

```bash
# Install core dependencies only
pip install PyPDF2 numpy

# Run the application
python main.py
```

### Option 2: Full Setup (Recommended)

```bash
# Install all dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Command Line Options

```bash
python main.py --help

Options:
  --port PORT        Port to run the server on (default: 8080)
  --no-browser       Don't open browser automatically
  --basic-mode       Force basic mode (no ML models)
```

## 📁 Project Structure

```
rag-chatbot/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── document_processor.py  # Document processing logic
│   ├── rag_system.py          # Core RAG implementation
│   └── web_interface.py       # HTTP server and UI
├── config/                 # Configuration files
│   └── settings.py
├── data/                   # Sample data files
│   ├── problem.pdf
│   ├── sample_data.json
│   └── sample_text.txt
└── docs/                   # Documentation
    └── README.md
```

## 🎯 How to Use

1. **Start the Application**
   ```bash
   python main.py
   ```

2. **Open Your Browser**
   - Go to `http://localhost:8080`
   - The browser should open automatically

3. **Upload a Document**
   - Click the upload area
   - Select a PDF, JSON, or TXT file
   - Click "Upload & Process"

4. **Ask Questions**
   - Type your question in the chat input
   - Press Enter or click "Ask"
   - Get intelligent answers with confidence scores

## 🔧 Architecture

### Core Components

1. **DocumentProcessor** (`src/document_processor.py`)
   - Handles PDF, JSON, and TXT file extraction
   - Implements intelligent text chunking
   - Cleans and normalizes text content

2. **RAGSystem** (`src/rag_system.py`)
   - Core retrieval-augmented generation logic
   - Supports both advanced (ML) and basic (keyword) modes
   - Manages embeddings and vector search

3. **WebInterface** (`src/web_interface.py`)
   - HTTP server implementation
   - Modern HTML/CSS/JavaScript frontend
   - RESTful API endpoints

### Operation Modes

- **Advanced Mode**: Uses SentenceTransformers + FAISS for semantic search
- **Basic Mode**: Uses keyword matching (fallback when ML libraries unavailable)

## 🛠️ Configuration

Edit `config/settings.py` to customize:

- Chunk size and overlap
- Confidence thresholds  
- Model settings
- UI preferences

## 📊 API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload and process documents
- `POST /ask` - Ask questions
- `GET /status` - Get system status

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   - Run with `--basic-mode` if ML libraries fail to install
   - Check Python version (3.8+ required)

2. **Port Already in Use**
   - Use `--port` to specify a different port
   - Kill existing Python processes

3. **File Upload Issues**
   - Ensure file formats are supported (.pdf, .json, .txt)
   - Check file size limits (50MB max)

### Debug Mode

```bash
# Run with verbose output
python main.py --basic-mode --no-browser --port 8081
```

## 🤝 Contributing

1. Follow the modular architecture
2. Add type hints to all functions
3. Include docstrings for all classes and methods
4. Test both advanced and basic modes

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ for intelligent document processing**
