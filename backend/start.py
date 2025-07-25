#!/usr/bin/env python3
"""
Startup script for Voice Shield: AI-Powered Aggression Detector
Handles both backend server and optional frontend serving
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'flask', 'flask_socketio', 'numpy', 'websockets'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def start_backend_server():
    """Start the Flask backend server"""
    print("🚀 Starting AI backend server...")
    try:
        # Import and run the server
        from server import run_server
        run_server()
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting backend server: {e}")
        return False
    return True

def start_frontend_server():
    """Start a simple HTTP server for the frontend"""
    print("🌐 Starting frontend server...")
    try:
        import http.server
        import socketserver
        
        PORT = 8000
        Handler = http.server.SimpleHTTPRequestHandler
        
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"📁 Frontend server running at http://localhost:{PORT}")
            print("📄 Open index.html in your browser or visit the URL above")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Error starting frontend server: {e}")

def open_browser():
    """Open the application in the default browser"""
    time.sleep(3)  # Wait for servers to start
    try:
        webbrowser.open('http://localhost:8000')
        print("🌐 Opened application in browser")
    except Exception as e:
        print(f"⚠️ Could not open browser automatically: {e}")
        print("Please manually open http://localhost:8000 in your browser")

def main():
    """Main startup function"""
    print("🎯 Voice Shield: AI-Powered Aggression Detector")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('server.py').exists():
        print("❌ Error: server.py not found in current directory")
        print("Please run this script from the project root directory")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("\n📋 Starting services...")
    
    # Start frontend server in a separate thread
    frontend_thread = threading.Thread(target=start_frontend_server, daemon=True)
    frontend_thread.start()
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start backend server (main thread)
    start_backend_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Shutting down Voice Shield...")
        sys.exit(0) 