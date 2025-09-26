"""
Configuration settings for the RAG chatbot
"""

# Document processing settings
CHUNK_SIZE = 500  # Maximum words per chunk
CHUNK_OVERLAP = 50  # Overlapping words between chunks

# Supported file formats
SUPPORTED_FORMATS = ['.pdf', '.json', '.txt']

# Web interface settings
DEFAULT_PORT = 8080
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# RAG system settings
DEFAULT_RETRIEVAL_K = 5  # Number of chunks to retrieve
CONFIDENCE_THRESHOLDS = {
    'high': 0.6,
    'medium': 0.3,
    'low': 0.0
}

# Advanced model settings
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
USE_FAISS = True

# UI settings
CHAT_HISTORY_LIMIT = 100
AUTO_CLEAR_CHAT = False
