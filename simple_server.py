#!/usr/bin/env python3
"""
Simple HTTP Server for HyperVision AI Platform
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def do_GET(self):
        # Serve the main platform page for root requests
        if self.path == '/':
            self.path = '/hypervision_arbitrage_platform.html'
        return super().do_GET()
    
    def end_headers(self):
        # Add CORS headers and proper content types
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        
        # Set proper content types
        if self.path.endswith('.js'):
            self.send_header('Content-Type', 'application/javascript')
        elif self.path.endswith('.html'):
            self.send_header('Content-Type', 'text/html')
        elif self.path.endswith('.css'):
            self.send_header('Content-Type', 'text/css')
        
        super().end_headers()

def main():
    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    print(f"🌐 Starting HyperVision AI Platform on port {port}")
    print(f"📂 Serving from: {Path(__file__).parent}")
    print(f"🚀 Platform will be accessible at: http://0.0.0.0:{port}")
    
    with socketserver.TCPServer(("0.0.0.0", port), CustomHandler) as httpd:
        print(f"✅ Server running on port {port}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()