#!/usr/bin/env python3
"""
Server runner for Customer Churn Prediction System
Optimized for Mac M1
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

class ServerManager:
    def __init__(self):
        self.django_process = None
        self.streamlit_process = None
        self.running = True
    
    def check_ports(self):
        """Check if ports are available"""
        import socket
        
        def is_port_open(port):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        
        if is_port_open(8000):
            print("⚠️  Port 8000 is already in use")
            return False
        
        if is_port_open(8501):
            print("⚠️  Port 8501 is already in use")
            return False
        
        return True
    
    def kill_existing_servers(self):
        """Kill any existing Django or Streamlit servers"""
        try:
            # Kill Django servers
            subprocess.run(['pkill', '-f', 'manage.py runserver'], 
                         capture_output=True, check=False)
            
            # Kill Streamlit servers
            subprocess.run(['pkill', '-f', 'streamlit'], 
                         capture_output=True, check=False)
            
            time.sleep(2)  # Wait for processes to die
        except:
            pass
    
    def start_django(self):
        """Start Django server"""
        print("🚀 Starting Django server...")
        
        django_dir = Path("django_backend")
        if not django_dir.exists():
            print("❌ Django backend directory not found")
            return False
        
        try:
            # Set environment variables for Mac M1
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path.cwd())
            env['DJANGO_SETTINGS_MODULE'] = 'django_backend.settings'
            
            self.django_process = subprocess.Popen(
                [sys.executable, 'manage.py', 'runserver', '127.0.0.1:8000'],
                cwd=django_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait a bit and check if it started
            time.sleep(3)
            if self.django_process.poll() is None:
                print("✅ Django server started on http://127.0.0.1:8000")
                return True
            else:
                stdout, stderr = self.django_process.communicate()
                print(f"❌ Django server failed to start: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting Django: {e}")
            return False
    
    def start_streamlit(self):
        """Start Streamlit server"""
        print("🚀 Starting Streamlit server...")
        
        if not Path("streamlit_app.py").exists():
            print("❌ Streamlit app file not found")
            return False
        
        try:
            # Set environment variables for Mac M1
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path.cwd())
            
            self.streamlit_process = subprocess.Popen([
                sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
                '--server.port', '8501',
                '--server.address', '127.0.0.1',
                '--server.headless', 'true',
                '--browser.gatherUsageStats', 'false'
            ], 
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
            )
            
            # Wait a bit and check if it started
            time.sleep(5)
            if self.streamlit_process.poll() is None:
                print("✅ Streamlit server started on http://127.0.0.1:8501")
                return True
            else:
                stdout, stderr = self.streamlit_process.communicate()
                print(f"❌ Streamlit server failed to start: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting Streamlit: {e}")
            return False
    
    def monitor_servers(self):
        """Monitor server processes"""
        while self.running:
            time.sleep(5)
            
            # Check Django
            if self.django_process and self.django_process.poll() is not None:
                print("⚠️  Django server stopped unexpectedly")
                self.running = False
                break
            
            # Check Streamlit
            if self.streamlit_process and self.streamlit_process.poll() is not None:
                print("⚠️  Streamlit server stopped unexpectedly")
                self.running = False
                break
    
    def stop_servers(self):
        """Stop all servers"""
        print("\n🛑 Stopping servers...")
        self.running = False
        
        if self.django_process:
            self.django_process.terminate()
            try:
                self.django_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.django_process.kill()
        
        if self.streamlit_process:
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
        
        print("✅ Servers stopped")
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C"""
        self.stop_servers()
        sys.exit(0)
    
    def run(self):
        """Main run method"""
        print("🚀 Customer Churn Prediction - Server Manager")
        print("=" * 50)
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Kill existing servers
        self.kill_existing_servers()
        
        # Check ports
        if not self.check_ports():
            print("❌ Ports not available. Trying to free them...")
            self.kill_existing_servers()
            time.sleep(2)
        
        # Start Django
        if not self.start_django():
            print("❌ Failed to start Django server")
            return False
        
        # Start Streamlit
        if not self.start_streamlit():
            print("❌ Failed to start Streamlit server")
            self.stop_servers()
            return False
        
        print("\n🎉 Both servers are running!")
        print("=" * 50)
        print("🌐 Web Interface: http://127.0.0.1:8501")
        print("🔗 API Endpoint: http://127.0.0.1:8000")
        print("\nPress Ctrl+C to stop servers")
        print("=" * 50)
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=self.monitor_servers)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        self.stop_servers()
        return True

def main():
    """Main function"""
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected")
        print("Please activate virtual environment first:")
        print("source venv/bin/activate")
        return False
    
    manager = ServerManager()
    return manager.run()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
