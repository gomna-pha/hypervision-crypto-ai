#!/usr/bin/env python3
import http.server
import socketserver
import sys

PORT = 8080
Handler = http.server.SimpleHTTPRequestHandler

try:
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"Server running at http://0.0.0.0:{PORT}/")
        print("Press Ctrl+C to stop the server")
        sys.stdout.flush()
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
