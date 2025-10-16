#!/usr/bin/env python3
"""
START VALIDATION SERVER
=======================
Starts the real-time validation server as a background service
"""

import subprocess
import sys
import os
import signal
import time
from pathlib import Path

def start_validation_server():
    """Start the validation server in the background"""
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Path to the validation server
    validator_path = script_dir / 'platform_validator.py'
    
    if not validator_path.exists():
        print(f"❌ Validation server not found at {validator_path}")
        return False
    
    try:
        # Start the validation server as a background process
        process = subprocess.Popen([
            sys.executable, str(validator_path)
        ], 
        cwd=str(script_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ Validation server started with PID: {process.pid}")
            
            # Save PID for later cleanup
            with open(script_dir / 'validation_server.pid', 'w') as f:
                f.write(str(process.pid))
            
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Validation server failed to start")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting validation server: {e}")
        return False

def stop_validation_server():
    """Stop the validation server"""
    
    script_dir = Path(__file__).parent.absolute()
    pid_file = script_dir / 'validation_server.pid'
    
    if not pid_file.exists():
        print("ℹ️ Validation server PID file not found")
        return True
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Try to terminate the process
        os.kill(pid, signal.SIGTERM)
        
        # Wait a bit and check if it's still running
        time.sleep(1)
        
        try:
            os.kill(pid, 0)  # Check if process exists
            # If we get here, process is still running, force kill
            os.kill(pid, signal.SIGKILL)
            print(f"🔪 Force killed validation server PID: {pid}")
        except ProcessLookupError:
            print(f"✅ Validation server stopped PID: {pid}")
        
        # Remove PID file
        pid_file.unlink()
        return True
        
    except Exception as e:
        print(f"❌ Error stopping validation server: {e}")
        return False

def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'start':
            success = start_validation_server()
            sys.exit(0 if success else 1)
            
        elif command == 'stop':
            success = stop_validation_server()
            sys.exit(0 if success else 1)
            
        elif command == 'restart':
            print("🔄 Restarting validation server...")
            stop_validation_server()
            time.sleep(1)
            success = start_validation_server()
            sys.exit(0 if success else 1)
            
        elif command == 'status':
            script_dir = Path(__file__).parent.absolute()
            pid_file = script_dir / 'validation_server.pid'
            
            if not pid_file.exists():
                print("❌ Validation server is not running")
                sys.exit(1)
            
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                os.kill(pid, 0)  # Check if process exists
                print(f"✅ Validation server is running with PID: {pid}")
                sys.exit(0)
                
            except ProcessLookupError:
                print("❌ Validation server process not found")
                pid_file.unlink()  # Clean up stale PID file
                sys.exit(1)
            except Exception as e:
                print(f"❌ Error checking validation server: {e}")
                sys.exit(1)
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Usage: python start_validation_server.py [start|stop|restart|status]")
            sys.exit(1)
    
    else:
        # Default: start the server
        success = start_validation_server()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()