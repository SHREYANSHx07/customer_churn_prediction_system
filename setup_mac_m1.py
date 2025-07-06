#!/usr/bin/env python3
"""
Mac M1 Setup Script for Customer Churn Prediction System
Handles all Mac M1 specific compatibility issues
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e.stderr}")
        return False

def check_system():
    """Check system requirements"""
    print("🖥️  System Check for Mac M1")
    print("=" * 40)
    
    # Check if running on Mac
    if platform.system() != 'Darwin':
        print("⚠️  Warning: This script is optimized for macOS")
    
    # Check if M1/M2 Mac
    try:
        result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
        if 'arm64' in result.stdout:
            print("✅ Apple Silicon (M1/M2) detected")
        else:
            print("ℹ️  Intel Mac detected")
    except:
        pass
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor} - Compatible")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} - Need 3.8+")
        return False
    
    return True

def setup_environment():
    """Setup virtual environment with Mac M1 optimizations"""
    print("\n🔧 Setting up Virtual Environment")
    print("=" * 40)
    
    # Remove existing venv
    if os.path.exists('venv'):
        run_command('rm -rf venv', 'Removing old virtual environment')
    
    # Create new venv
    if not run_command('python3 -m venv venv', 'Creating virtual environment'):
        return False
    
    # Activate and upgrade pip
    activate_cmd = 'source venv/bin/activate'
    
    commands = [
        (f'{activate_cmd} && pip install --upgrade pip', 'Upgrading pip'),
        (f'{activate_cmd} && pip install wheel setuptools', 'Installing build tools'),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    return True

def install_dependencies():
    """Install dependencies with Mac M1 specific handling"""
    print("\n📦 Installing Dependencies")
    print("=" * 40)
    
    activate_cmd = 'source venv/bin/activate'
    
    # Mac M1 specific scikit-learn installation
    commands = [
        # Install numpy first (required for scikit-learn)
        (f'{activate_cmd} && pip install "numpy==1.24.3"', 'Installing NumPy'),
        
        # Install scikit-learn with specific version for M1 compatibility
        (f'{activate_cmd} && pip install --only-binary=all "scikit-learn==1.3.2"', 'Installing scikit-learn (M1 optimized)'),
        
        # Install other ML dependencies
        (f'{activate_cmd} && pip install "pandas==2.1.3"', 'Installing Pandas'),
        
        # Install web framework dependencies
        (f'{activate_cmd} && pip install "Django==4.2.7" "djangorestframework==3.14.0"', 'Installing Django'),
        
        # Install remaining requirements
        (f'{activate_cmd} && pip install -r requirements.txt', 'Installing remaining packages'),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            # Try alternative installation for scikit-learn
            if 'scikit-learn' in cmd:
                print("🔄 Trying alternative scikit-learn installation...")
                alt_cmd = f'{activate_cmd} && pip install "scikit-learn==1.3.0"'
                if not run_command(alt_cmd, 'Installing scikit-learn (alternative)'):
                    return False
            else:
                return False
    
    return True

def setup_django():
    """Setup Django backend"""
    print("\n🗄️  Setting up Django Backend")
    print("=" * 40)
    
    activate_cmd = 'source venv/bin/activate'
    
    commands = [
        (f'{activate_cmd} && cd django_backend && python manage.py makemigrations churn_app', 'Creating migrations'),
        (f'{activate_cmd} && cd django_backend && python manage.py migrate', 'Applying migrations'),
        (f'{activate_cmd} && cd django_backend && python manage.py collectstatic --noinput', 'Collecting static files'),
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc)  # Don't fail on collectstatic
    
    return True

def verify_installation():
    """Verify the installation"""
    print("\n✅ Verifying Installation")
    print("=" * 40)
    
    activate_cmd = 'source venv/bin/activate'
    
    # Test imports
    test_script = '''
import sys
print(f"Python: {sys.version}")

try:
    import django
    print(f"✅ Django: {django.VERSION}")
except ImportError as e:
    print(f"❌ Django: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn: {e}")

try:
    import streamlit as st
    print(f"✅ Streamlit: {st.__version__}")
except ImportError as e:
    print(f"❌ Streamlit: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")
'''
    
    with open('test_imports.py', 'w') as f:
        f.write(test_script)
    
    success = run_command(f'{activate_cmd} && python test_imports.py', 'Testing package imports')
    
    # Clean up
    if os.path.exists('test_imports.py'):
        os.remove('test_imports.py')
    
    return success

def main():
    """Main setup function"""
    print("🚀 Customer Churn Prediction - Mac M1 Setup")
    print("=" * 50)
    
    if not check_system():
        print("❌ System check failed")
        return False
    
    if not setup_environment():
        print("❌ Environment setup failed")
        return False
    
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return False
    
    if not setup_django():
        print("❌ Django setup failed")
        return False
    
    if not verify_installation():
        print("❌ Installation verification failed")
        return False
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Activate environment: source venv/bin/activate")
    print("2. Start servers: python run_servers.py")
    print("3. Open browser: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    main()
