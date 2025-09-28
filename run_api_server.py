
import sys
sys.path.insert(0, '/home/user/webapp')

import asyncio
import uvicorn
from hypervision_api_server import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        ws_ping_interval=10,
        ws_ping_timeout=5,
        ws_max_size=10**6
    )
