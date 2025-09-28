
import http.server
import socketserver
import os

os.chdir('/home/user/webapp')

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        if self.path == '/':
            self.path = '/opportunities_live.html'
        return super().do_GET()

with socketserver.TCPServer(("", 8081), MyHandler) as httpd:
    print("Opportunities server running on port 8081")
    httpd.serve_forever()
