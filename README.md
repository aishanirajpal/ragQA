# 🤖 RAG Document QA Chatbot

A **clean, modular, and professional** implementation of a Retrieval-Augmented Generation (RAG) system for intelligent document question answering.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

## ✨ Features

- **📄 Multi-Format Support**: PDF, JSON, and TXT files
- **🧠 Dual Mode Operation**: Advanced (ML-based) and Basic (keyword-based)
- **🎨 Modern Web Interface**: Beautiful, responsive design
- **🔍 Intelligent Search**: Semantic similarity or keyword matching
- **📊 Confidence Scoring**: Know how reliable the answers are
- **🏗️ Modular Architecture**: Clean, maintainable code structure
- **⚡ Fast Performance**: Optimized for speed and efficiency

## 🚀 Quick Start

### Minimum Setup (Core Features Only)
```bash
# Install core dependencies
pip install PyPDF2 numpy

# Run the application
python main.py
```

### Full Setup (Recommended)
```bash
# Install all dependencies for advanced features
pip install -r requirements.txt

# Run the application  
python main.py
```

### Access the Interface
- **URL**: http://localhost:8080
- **Browser**: Opens automatically
- **Usage**: Upload documents and start asking questions!

## 🧠 Gemini Integration

For the highest quality answers, this system can be integrated with the **Google Gemini API** using LangChain.

### Setup

1.  **Install extra dependencies**:
    ```bash
    pip install langchain langchain-google-genai python-dotenv
    ```
    (These are also included in `requirements.txt`)

2.  **Get a Gemini API Key**:
    - Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to get your free API key.

3.  **Create a `.env` file**:
    - In the project's root directory, create a file named `.env`.
    - Add your API key to it, like this:
      ```
      GOOGLE_API_KEY="your_gemini_api_key_here"
      ```

### How it Works

- If a valid `GOOGLE_API_KEY` is found in the `.env` file, the chatbot will automatically switch to **Generative Mode**.
- In this mode, the retrieved document chunks are sent to the Gemini model along with your question to generate a more natural, conversational answer.
- If no key is found, it gracefully falls back to the default extractive method.

## 📁 Clean Project Structure

```
📦 rag-chatbot/
├── 🚀 main.py                 # Main application entry point
├── 📋 requirements.txt        # Python dependencies  
├── 📂 src/                    # Source code modules
│   ├── 📄 document_processor.py  # Document processing logic
│   ├── 🧠 rag_system.py          # Core RAG implementation
│   └── 🌐 web_interface.py       # HTTP server and UI
├── ⚙️ config/                # Configuration files
│   └── settings.py
├── 📊 data/                   # Sample data files
│   ├── problem.pdf
│   ├── sample_data.json
│   └── sample_text.txt
└── 📚 docs/                   # Documentation
    └── README.md
```

## 🎯 Usage

1. **Start Application**
   ```bash
   python main.py --help  # See all options
   ```

2. **Upload Document**
   - Drag & drop or click to select
   - Supports PDF, JSON, TXT files
   - Automatic processing and indexing

3. **Ask Questions**
   - Natural language queries
   - Get intelligent answers
   - View confidence scores and sources

## 🔧 Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --port PORT        Port to run server on (default: 8080)
  --no-browser       Don't open browser automatically  
  --basic-mode       Force basic mode (no ML models)
```

## 🏗️ Architecture

### Modular Design
- **DocumentProcessor**: Handles file extraction and chunking
- **RAGSystem**: Core retrieval and generation logic
- **WebInterface**: Modern HTTP server with beautiful UI

### Operation Modes
- **✨ Generative (Gemini)**: LangChain + Gemini for conversational answers
- **🚀 Advanced**: SentenceTransformers + FAISS for semantic search
- **🔧 Basic**: Keyword matching (fallback when ML unavailable)

## 📊 Technical Details

### Supported Formats
- **PDF**: Full text extraction with PyPDF2
- **JSON**: Recursive key-value extraction  
- **TXT**: UTF-8 and Latin-1 encoding support

### Performance
- **Chunking**: Intelligent text segmentation with overlap
- **Search**: Vector similarity or keyword matching
- **Caching**: Efficient embedding storage with FAISS
- **Generation**: Extractive (keyword-based sentence selection) or Generative (with Gemini)

## 🛠️ Configuration

Edit `config/settings.py` for customization:
- Chunk size and overlap settings
- Confidence thresholds
- Model parameters (Gemini model, etc.)
- UI preferences

## 🔍 Troubleshooting

### Common Issues
1. **Import errors**: Use `--basic-mode` flag
2. **Port conflicts**: Use `--port` to change port
3. **Memory issues**: Reduce chunk size in config

### Debug Mode
```bash
python main.py --basic-mode --no-browser --port 8081
```

### Gemini Errors
- Ensure your API key is correct and has quota. Check the console for detailed error messages.

## 🤝 Contributing

This project follows clean code principles:
- Modular architecture
- Type hints throughout
- Comprehensive docstrings
- Error handling
- Configuration management

## 📄 License

MIT License - see LICENSE file for details.

---

**🎉 Built with ❤️ for intelligent document processing**

*Clean • Modular • Professional • Production-Ready*
