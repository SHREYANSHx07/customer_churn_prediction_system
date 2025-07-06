#!/bin/bash

# Mac M1 Quick Start Script for Customer Churn Prediction
# This script handles all Mac M1 specific setup and startup

echo "🚀 Customer Churn Prediction - Mac M1 Quick Start"
echo "=================================================="

# Check if we're on Mac
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is optimized for macOS"
fi

# Check if we're on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo "✅ Apple Silicon (M1/M2) detected"
else
    echo "ℹ️  Intel Mac detected"
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✅ Python $PYTHON_VERSION found"
else
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/lib/python*/site-packages/django/__init__.py" ]; then
    echo "📦 Installing dependencies..."
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install numpy first (required for scikit-learn on M1)
    pip install "numpy==1.24.3"
    
    # Install scikit-learn with M1 optimization
    pip install --only-binary=all "scikit-learn==1.3.2" || pip install "scikit-learn==1.3.0"
    
    # Install remaining requirements
    pip install -r requirements.txt
    
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Setup Django database
if [ ! -f "django_backend/db.sqlite3" ]; then
    echo "🗄️  Setting up Django database..."
    cd django_backend
    python manage.py makemigrations churn_app
    python manage.py migrate
    cd ..
    echo "✅ Database setup complete"
else
    echo "✅ Database already exists"
fi

# Kill any existing servers
echo "🧹 Cleaning up existing servers..."
pkill -f "manage.py runserver" 2>/dev/null || true
pkill -f "streamlit" 2>/dev/null || true
sleep 2

# Start servers
echo "🚀 Starting servers..."
python run_servers.py &
SERVER_PID=$!

# Wait for servers to start
echo "⏳ Waiting for servers to start..."
sleep 10

# Check if servers are running
if curl -s http://127.0.0.1:8000/api/customers/ >/dev/null 2>&1; then
    echo "✅ Django server running on http://127.0.0.1:8000"
else
    echo "❌ Django server failed to start"
fi

if curl -s http://127.0.0.1:8501 >/dev/null 2>&1; then
    echo "✅ Streamlit server running on http://127.0.0.1:8501"
else
    echo "❌ Streamlit server failed to start"
fi

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo "🌐 Web Interface: http://127.0.0.1:8501"
echo "🔗 API Endpoint: http://127.0.0.1:8000"
echo ""
echo "Press Ctrl+C to stop servers"
echo "=================="

# Keep script running
wait $SERVER_PID
