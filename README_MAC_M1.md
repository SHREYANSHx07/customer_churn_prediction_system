# Customer Churn Prediction System - Mac M1 Optimized

## 🚀 Quick Start for Mac M1/M2

### Prerequisites
- macOS with Apple Silicon (M1/M2) or Intel
- Python 3.8 or higher
- 2GB free disk space

### One-Command Setup
\`\`\`bash
chmod +x mac_m1_quick_start.sh
./mac_m1_quick_start.sh
\`\`\`

### Manual Setup

1. **Setup Environment**
\`\`\`bash
python3 setup_mac_m1.py
\`\`\`

2. **Start Servers**
\`\`\`bash
source venv/bin/activate
python run_servers.py
\`\`\`

3. **Test System**
\`\`\`bash
python test_system_mac_m1.py
\`\`\`

### Access Points
- **Web Interface**: http://127.0.0.1:8501
- **API Endpoint**: http://127.0.0.1:8000

### Mac M1 Specific Optimizations

1. **scikit-learn Compatibility**
   - Uses version 1.3.2 with binary-only installation
   - Fallback to version 1.3.0 if needed

2. **NumPy Optimization**
   - Version 1.24.3 for M1 compatibility
   - Installed before scikit-learn

3. **Server Configuration**
   - Single-threaded RandomForest for stability
   - Optimized memory usage
   - Mac-specific process management

### Troubleshooting

#### scikit-learn Installation Issues
\`\`\`bash
pip install --only-binary=all "scikit-learn==1.3.2"
# OR
pip install "scikit-learn==1.3.0"
\`\`\`

#### Port Conflicts
\`\`\`bash
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9
\`\`\`

#### Virtual Environment Issues
\`\`\`bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

### Features

- ✅ CSV data upload and processing
- ✅ Machine learning model training
- ✅ Churn prediction and risk assessment
- ✅ Interactive web dashboard
- ✅ Analytics and visualizations
- ✅ Data export functionality
- ✅ Mac M1/M2 optimized performance

### System Requirements

- **Minimum**: 4GB RAM, 2GB storage
- **Recommended**: 8GB RAM, 5GB storage
- **Python**: 3.8 - 3.11 (3.13 supported with specific setup)

### Support

For issues specific to Mac M1/M2:
1. Check Python version compatibility
2. Verify virtual environment activation
3. Run system test: `python test_system_mac_m1.py`
4. Check server logs for detailed error messages

### Performance Notes

- Model training: 30-60 seconds for 1000 customers
- Predictions: 5-10 seconds for 1000 customers
- Data upload: Supports up to 10MB CSV files
- Concurrent users: Optimized for single-user development use
