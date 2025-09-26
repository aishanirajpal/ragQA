#!/usr/bin/env python3
"""
Quick launcher for the RAG Document QA Chatbot
"""

import subprocess
import sys
import os

def main():
    """Launch the RAG chatbot with smart defaults"""
    
    print("🤖 RAG Document QA Chatbot - Quick Launcher")
    print("=" * 50)
    
    # Check if main.py exists
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found in current directory")
        return 1
    
    # Try to run with advanced features first, fallback to basic mode
    try:
        print("🚀 Attempting to start with advanced features...")
        result = subprocess.run([sys.executable, "main.py"], check=True)
        return result.returncode
    except subprocess.CalledProcessError:
        print("⚠️  Advanced mode failed, trying basic mode...")
        try:
            result = subprocess.run([sys.executable, "main.py", "--basic-mode"], check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"❌ Error starting application: {e}")
            return 1
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        return 0

if __name__ == "__main__":
    sys.exit(main())
