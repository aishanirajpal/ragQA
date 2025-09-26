"""
Web Interface Module
HTTP server and HTML interface for the RAG chatbot
"""

import json
import threading
import webbrowser
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional


class ChatbotHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the chatbot web interface"""
    
    def __init__(self, rag_system, *args, **kwargs):
        self.rag_system = rag_system
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(self._get_html_template().encode())
        elif self.path == '/status':
            self._handle_status()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/upload':
            self._handle_upload()
        elif self.path == '/ask':
            self._handle_question()
        elif self.path == '/clear':
            self._handle_clear()
        else:
            self.send_error(404)
    
    def _handle_upload(self):
        """Handle multiple file uploads with robust multipart parsing"""
        try:
            # Get content type and boundary
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                raise ValueError("Invalid content type for file upload")
            
            # Extract boundary
            boundary = None
            for part in content_type.split(';'):
                if 'boundary=' in part:
                    boundary = part.split('boundary=')[1].strip()
                    break
            
            if not boundary:
                raise ValueError("No boundary found in multipart data")
            
            # Read the raw post data
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                raise ValueError("No content in upload")
            
            raw_data = self.rfile.read(content_length)
            
            # Parse multipart manually (more reliable for binary data)
            boundary_bytes = f'--{boundary}'.encode()
            parts = raw_data.split(boundary_bytes)
            
            files_processed = 0
            for part in parts:
                if b'Content-Disposition' in part and b'filename=' in part:
                    # Find the header/content boundary
                    header_end = part.find(b'\r\n\r\n')
                    if header_end == -1:
                        header_end = part.find(b'\n\n')
                        content_start = header_end + 2
                    else:
                        content_start = header_end + 4
                    
                    if header_end > 0 and content_start < len(part):
                        headers = part[:header_end].decode('utf-8', errors='ignore')
                        
                        # Extract filename with better parsing
                        import re
                        filename_match = re.search(r'filename="([^"]*)"', headers)
                        if filename_match:
                            filename = filename_match.group(1)
                        else:
                            # Try without quotes
                            filename_match = re.search(r'filename=([^\s;]+)', headers)
                            if filename_match:
                                filename = filename_match.group(1)
                        
                        # Extract file content (keep as bytes)
                        file_content = part[content_start:]
                        
                        # Clean up trailing boundary markers and whitespace
                        while file_content.endswith(b'\r\n') or file_content.endswith(b'\n') or file_content.endswith(b'\r'):
                            if file_content.endswith(b'\r\n'):
                                file_content = file_content[:-2]
                            elif file_content.endswith(b'\n'):
                                file_content = file_content[:-1]
                            elif file_content.endswith(b'\r'):
                                file_content = file_content[:-1]
                        
                        if filename and file_content:
                            print(f"📄 Processing upload: {filename} ({len(file_content)} bytes)")
                            self.rag_system.load_document(filename, file_content)
                            files_processed += 1
            
            if files_processed == 0:
                raise ValueError("No valid files found in the upload payload.")

            doc_info = self.rag_system.get_document_info()
            
            response = {
                "success": True,
                "message": f"{files_processed} document(s) processed successfully!",
                "info": doc_info
            }
            
        except Exception as e:
            response = {"success": False, "message": str(e)}
            print(f"Upload error: {e}")
            import traceback
            traceback.print_exc()
        
        self._send_json_response(response)
    
    def _handle_clear(self):
        """Handles request to clear all documents."""
        try:
            self.rag_system.clear_documents()
            response = {"success": True, "message": "All documents cleared."}
        except Exception as e:
            response = {"success": False, "message": str(e)}
            print(f"Clear error: {e}")
        self._send_json_response(response)

    def _handle_question(self):
        """Handle question answering"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            question = data.get('question', '')
            if not question:
                raise ValueError("No question provided")
            
            result = self.rag_system.answer_question(question)
            
        except Exception as e:
            result = {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "mode": "error"
            }
            print(f"Question error: {e}")
        
        self._send_json_response(result)
    
    def _handle_status(self):
        """Handle status request"""
        doc_info = self.rag_system.get_document_info()
        self._send_json_response(doc_info)
    
    def _send_json_response(self, data: dict):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _get_html_template(self) -> str:
        """Get the HTML template for the web interface"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 RAG Based QA Chatbot</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            padding: 30px;
        }
        
        .upload-section, .chat-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
        }
        
        .section-title {
            font-size: 1.3em;
            margin-bottom: 20px;
            color: #333;
            font-weight: 500;
        }
        
        .file-upload {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            background: white;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .file-upload:hover {
            border-color: #5a6fd8;
            background: #f0f4ff;
        }
        
        .file-upload input[type="file"] {
            display: none;
        }
        
        .file-upload label {
            display: block;
            cursor: pointer;
            font-size: 1.1em;
            color: #667eea;
            font-weight: 500;
        }
        
        .file-upload .upload-icon {
            font-size: 3em;
            margin-bottom: 15px;
            display: block;
        }
        
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            background: #5a6fd8;
            transform: translateY(-2px);
        }
        
        .btn-secondary {
            background: #6c757d;
        }
        
        .btn-secondary:hover {
            background: #5a6268;
        }
        
        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-weight: 500;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .status.info {
            background: #cce5ff;
            color: #004085;
            border: 1px solid #b3d7ff;
        }
        
        .chat-box {
            height: 450px;
            overflow-y: auto;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            background: white;
            margin-bottom: 20px;
        }
        
        .message {
            margin: 20px 0;
            padding: 15px 20px;
            border-radius: 15px;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .message.user {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            color: #1976d2;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }
        
        .message.bot {
            background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
            color: #388e3c;
            margin-right: auto;
            border-bottom-left-radius: 5px;
        }
        
        .input-area {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .input-area input {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s ease;
        }
        
        .input-area input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        .button-group.horizontal {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 15px;
        }

        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        .mode-indicator {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: 500;
            margin-left: 10px;
        }
        
        .mode-generative {
            background: #e6e6fa;
            color: #483d8b;
        }

        .mode-advanced {
            background: #e8f5e8;
            color: #2e7d32;
        }
        
        .mode-basic {
            background: #fff3e0;
            color: #f57c00;
        }
        
        #docInfo {
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            font-size: 0.9em;
            border: 1px solid #e9ecef;
        }

        .doc-info-title {
            margin-bottom: 10px;
            font-weight: 500;
            color: #333;
        }

        #docList {
            list-style-type: none;
            padding-left: 0;
        }

        #docList li {
            padding: 5px;
            border-bottom: 1px solid #f1f3f5;
        }
        
        #docList li:last-child {
            border-bottom: none;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 RAG Document QA Chatbot</h1>
            <p>Upload your documents and ask questions</p>
        </div>
        
        <div class="main-content">
            <div class="upload-section">
                <h3 class="section-title">📄 Document Collection</h3>
                
                <div class="file-upload">
                    <span class="upload-icon">📁</span>
                    <label for="fileInput">
                        Click to select or drag & drop files
                        <br><small>Supports PDF, JSON, TXT files</small>
                    </label>
                    <input type="file" id="fileInput" accept=".pdf,.json,.txt" multiple>
                </div>
                
                <div class="button-group horizontal">
                    <button class="btn" onclick="uploadFile()">🚀 Upload & Process</button>
                    <button class="btn btn-secondary" onclick="clearDocuments()">🗑️ Clear All</button>
                </div>

                <div id="uploadStatus" style="display: none;"></div>
                
                <div id="docInfo">
                    <h4 class="doc-info-title">Loaded Documents:</h4>
                    <ul id="docList">
                        <li>No documents loaded.</li>
                    </ul>
                </div>
            </div>
            
            <div class="chat-section">
                <h3 class="section-title">💬 Chat Interface</h3>
                
                <div id="chatBox" class="chat-box">
                    <div class="message bot">
                        👋 Welcome! Upload a document to get started. I can answer questions about PDF, JSON, and TXT files using advanced AI techniques.
                    </div>
                </div>
                
                <div class="input-area">
                    <input type="text" id="questionInput" placeholder="Ask about your documents..." onkeypress="if(event.key==='Enter') askQuestion()">
                    <button class="btn" onclick="askQuestion()">🔍 Ask</button>
                </div>
                
                <div class="button-group horizontal">
                    <button class="btn btn-secondary" onclick="clearChat()">🗑️ Clear Chat</button>
                    <button class="btn btn-secondary" onclick="showSources()">📚 Show Sources</button>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🔧 Powered by advanced RAG (Retrieval-Augmented Generation) technology</p>
            <p>📊 Confidence: 🟢 High (>60%) • 🟡 Medium (30-60%) • 🔴 Low (<30%)</p>
        </div>
    </div>

    <script>
        let currentQuestion = '';
        
        // File upload handling
        document.getElementById('fileInput').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                const fileName = e.target.files[0].name;
                document.querySelector('.file-upload label').innerHTML = 
                    `📄 ${fileName} selected<br><small>Click "Upload & Process" to continue</small>`;
            }
        });
        
        function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const statusDiv = document.getElementById('uploadStatus');
            const files = fileInput.files;

            if (files.length === 0) {
                showStatus('Please select one or more files first.', 'error');
                return;
            }
            
            showStatus(`📤 Uploading ${files.length} file(s)...`, 'info');
            
            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('document', files[i]);
            }
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('✅ ' + data.message, 'success');
                    showDocumentInfo(data.info);
                    addMessage(`📄 ${data.message} You can now ask questions about the loaded documents.`, 'bot');
                } else {
                    showStatus('❌ ' + data.message, 'error');
                }
            })
            .catch(error => {
                showStatus('❌ Upload failed: ' + error.message, 'error');
                console.error('Upload error:', error);
            });
        }
        
        function askQuestion() {
            const questionInput = document.getElementById('questionInput');
            const question = questionInput.value.trim();
            
            if (!question) return;
            
            currentQuestion = question;
            addMessage('❓ ' + question, 'user');
            questionInput.value = '';
            
            const typingMsg = addMessage('🤔 Thinking...', 'bot');
            
            fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            })
            .then(response => response.json())
            .then(data => {
                typingMsg.remove();
                
                const confidence = data.confidence > 0.6 ? '🟢' : data.confidence > 0.3 ? '🟡' : '🔴';
                const modeIndicator = data.mode === 'generative' ?
                    '<span class="mode-indicator mode-generative">✨ Generative</span>' :
                    data.mode === 'advanced' ? 
                    '<span class="mode-indicator mode-advanced">Advanced</span>' : 
                    '<span class="mode-indicator mode-basic">Basic</span>';
                
                const answer = `🤖 ${data.answer}<br><br>${confidence} Confidence: ${(data.confidence * 100).toFixed(0)}% ${modeIndicator}`;
                addMessage(answer, 'bot');
            })
            .catch(error => {
                typingMsg.remove();
                addMessage('❌ Error: ' + error.message, 'bot');
                console.error('Ask error:', error);
            });
        }
        
        function addMessage(text, sender) {
            const chatBox = document.getElementById('chatBox');
            const message = document.createElement('div');
            message.className = `message ${sender}`;
            message.innerHTML = text.replace(/\\n/g, '<br>');
            chatBox.appendChild(message);
            chatBox.scrollTop = chatBox.scrollHeight;
            return message;
        }
        
        function clearChat() {
            const chatBox = document.getElementById('chatBox');
            chatBox.innerHTML = '<div class="message bot">👋 Chat cleared! Ask me anything about your loaded documents.</div>';
        }

        function clearDocuments() {
            showStatus('🗑️ Clearing all documents...', 'info');
            fetch('/clear', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('✅ All documents have been cleared.', 'success');
                    showDocumentInfo(null); // Clear the doc info panel
                    addMessage('🗑️ All documents have been cleared. Ready for new uploads!', 'bot');
                } else {
                    showStatus('❌ Error clearing documents: ' + data.message, 'error');
                }
            })
            .catch(error => {
                showStatus('❌ Network error while clearing documents.', 'error');
            });
        }
        
        function showSources() {
            if (!currentQuestion) {
                addMessage('📚 Please ask a question first to see sources.', 'bot');
                return;
            }
            
            fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: currentQuestion })
            })
            .then(response => response.json())
            .then(data => {
                if (data.sources && data.sources.length > 0) {
                    let sourcesText = '📚 <strong>Sources used for the last answer:</strong><br><br>';
                    data.sources.forEach((source, index) => {
                        sourcesText += `<strong>Source ${index + 1} from "${source.source}":</strong><br>${source.content.substring(0, 200)}...<br><br>`;
                    });
                    addMessage(sourcesText, 'bot');
                } else {
                    addMessage('📚 No sources available for the last question.', 'bot');
                }
            })
            .catch(error => {
                addMessage('❌ Error retrieving sources: ' + error.message, 'bot');
            });
        }
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
            statusDiv.style.display = 'block';
            
            if (type === 'success') {
                setTimeout(() => {
                    statusDiv.style.display = 'none';
                }, 3000);
            }
        }
        
        function showDocumentInfo(info) {
            const docList = document.getElementById('docList');
            const docInfoTitle = document.querySelector('.doc-info-title');

            if (!info || !info.doc_names || info.doc_names.length === 0) {
                docList.innerHTML = '<li>No documents loaded.</li>';
                docInfoTitle.textContent = 'Loaded Documents:';
                return;
            }

            // Update the title with the current count
            docInfoTitle.textContent = `Loaded Documents (${info.doc_names.length}):`;

            docList.innerHTML = ''; // Clear existing list to rebuild it
            info.doc_names.forEach(name => {
                const li = document.createElement('li');
                li.textContent = `📄 ${name}`;
                docList.appendChild(li);
            });
        }
    </script>
</body>
</html>
        '''
    
    def log_message(self, format, *args):
        """Override to reduce console spam"""
        if "POST" in format:
            print(f"📡 {format % args}")


class WebInterface:
    """Web interface manager for the RAG chatbot"""
    
    def __init__(self, rag_system, port: int = 8080):
        """
        Initialize web interface
        
        Args:
            rag_system: RAG system instance
            port: Port to run the server on
        """
        self.rag_system = rag_system
        self.port = port
        self.server: Optional[HTTPServer] = None
    
    def create_handler(self):
        """Create a request handler with the RAG system"""
        def handler(*args, **kwargs):
            return ChatbotHandler(self.rag_system, *args, **kwargs)
        return handler
    
    def start_server(self, open_browser: bool = True):
        """Start the web server"""
        handler_class = self.create_handler()
        self.server = HTTPServer(('localhost', self.port), handler_class)
        
        print(f"🚀 Starting web server at http://localhost:{self.port}")
        
        if open_browser:
            def open_browser_delayed():
                time.sleep(1.5)
                print("📱 Opening browser...")
                webbrowser.open(f'http://localhost:{self.port}')
            
            threading.Thread(target=open_browser_delayed, daemon=True).start()
        
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\\n🛑 Server stopped")
            self.stop_server()
    
    def stop_server(self):
        """Stop the web server"""
        if self.server:
            self.server.shutdown()
            self.server = None
