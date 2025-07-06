#!/usr/bin/env python3
"""
Complete system test for Mac M1
Tests all components and provides detailed feedback
"""

import requests
import time
import subprocess
import sys
import os
from pathlib import Path

class MacM1SystemTester:
    def __init__(self):
        self.django_url = "http://127.0.0.1:8000"
        self.streamlit_url = "http://127.0.0.1:8501"
        self.test_results = []
    
    def log_test(self, test_name, success, message=""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append({
            'test': test_name,
            'success': success,
            'message': message
        })
        print(f"{status} {test_name}: {message}")
    
    def test_python_environment(self):
        """Test Python environment and packages"""
        print("\n🐍 Testing Python Environment")
        print("-" * 40)
        
        # Test Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.log_test("Python Version", True, f"{python_version.major}.{python_version.minor}")
        else:
            self.log_test("Python Version", False, f"Need 3.8+, got {python_version.major}.{python_version.minor}")
        
        # Test virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        self.log_test("Virtual Environment", in_venv, "Active" if in_venv else "Not detected")
        
        # Test key packages
        packages = ['django', 'pandas', 'sklearn', 'streamlit', 'numpy']
        for package in packages:
            try:
                __import__(package)
                self.log_test(f"Package {package}", True, "Imported successfully")
            except ImportError as e:
                self.log_test(f"Package {package}", False, str(e))
    
    def test_file_structure(self):
        """Test project file structure"""
        print("\n📁 Testing File Structure")
        print("-" * 40)
        
        required_files = [
            'requirements.txt',
            'run_servers.py',
            'streamlit_app.py',
            'django_backend/manage.py',
            'django_backend/django_backend/settings.py',
            'django_backend/churn_app/models.py',
            'django_backend/churn_app/views.py',
        ]
        
        for file_path in required_files:
            exists = Path(file_path).exists()
            self.log_test(f"File {file_path}", exists, "Found" if exists else "Missing")
    
    def test_django_server(self):
        """Test Django server"""
        print("\n🗄️  Testing Django Server")
        print("-" * 40)
        
        try:
            response = requests.get(f"{self.django_url}/api/customers/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                total_customers = data.get('pagination', {}).get('total_count', 0)
                self.log_test("Django API", True, f"Responding, {total_customers} customers")
            else:
                self.log_test("Django API", False, f"HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.log_test("Django API", False, "Connection refused - server not running")
        except Exception as e:
            self.log_test("Django API", False, str(e))
    
    def test_streamlit_server(self):
        """Test Streamlit server"""
        print("\n🎨 Testing Streamlit Server")
        print("-" * 40)
        
        try:
            response = requests.get(self.streamlit_url, timeout=10)
            if response.status_code == 200:
                self.log_test("Streamlit UI", True, "Responding")
            else:
                self.log_test("Streamlit UI", False, f"HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.log_test("Streamlit UI", False, "Connection refused - server not running")
        except Exception as e:
            self.log_test("Streamlit UI", False, str(e))
    
    def test_data_operations(self):
        """Test data upload and processing"""
        print("\n📊 Testing Data Operations")
        print("-" * 40)
        
        # Test sample data generation
        try:
            response = requests.post(
                f"{self.django_url}/api/generate-sample-data/",
                json={"n_customers": 20},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                self.log_test("Sample Data Generation", True, f"Created {data.get('created_count', 0)} customers")
            else:
                self.log_test("Sample Data Generation", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Sample Data Generation", False, str(e))
        
        # Test data retrieval
        try:
            response = requests.get(f"{self.django_url}/api/customers/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                count = data.get('pagination', {}).get('total_count', 0)
                self.log_test("Data Retrieval", True, f"{count} customers retrieved")
            else:
                self.log_test("Data Retrieval", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Data Retrieval", False, str(e))
    
    def test_ml_operations(self):
        """Test ML model training and prediction"""
        print("\n🤖 Testing ML Operations")
        print("-" * 40)
        
        # Test model training
        try:
            response = requests.post(f"{self.django_url}/api/train-model/", timeout=120)
            if response.status_code == 200:
                data = response.json()
                accuracy = data.get('metrics', {}).get('accuracy', 0)
                self.log_test("Model Training", True, f"Accuracy: {accuracy:.3f}")
            else:
                error_msg = response.text[:100] if response.text else f"HTTP {response.status_code}"
                self.log_test("Model Training", False, error_msg)
        except Exception as e:
            self.log_test("Model Training", False, str(e))
        
        # Test predictions
        try:
            response = requests.post(f"{self.django_url}/api/predict/", timeout=60)
            if response.status_code == 200:
                data = response.json()
                total_pred = data.get('total_predictions', 0)
                high_risk = data.get('high_risk_count', 0)
                self.log_test("Predictions", True, f"{total_pred} predictions, {high_risk} high risk")
            else:
                error_msg = response.text[:100] if response.text else f"HTTP {response.status_code}"
                self.log_test("Predictions", False, error_msg)
        except Exception as e:
            self.log_test("Predictions", False, str(e))
    
    def test_analytics(self):
        """Test analytics endpoint"""
        print("\n📈 Testing Analytics")
        print("-" * 40)
        
        try:
            response = requests.get(f"{self.django_url}/api/analytics/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                total_customers = data.get('total_customers', 0)
                churn_rate = data.get('churn_rate', 0)
                self.log_test("Analytics", True, f"{total_customers} customers, {churn_rate}% churn rate")
            else:
                self.log_test("Analytics", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Analytics", False, str(e))
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result['success'])
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED!")
            print("Your Customer Churn Prediction System is fully operational on Mac M1!")
            print("\n📋 Next Steps:")
            print("1. Open browser: http://127.0.0.1:8501")
            print("2. Upload your CSV data or use sample data")
            print("3. Train model and generate predictions")
            print("4. Explore analytics and export results")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("Failed tests:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  ❌ {result['test']}: {result['message']}")
            
            print("\n🔧 Troubleshooting:")
            print("1. Make sure virtual environment is activated: source venv/bin/activate")
            print("2. Install dependencies: pip install -r requirements.txt")
            print("3. Start servers: python run_servers.py")
            print("4. Check Django migrations: cd django_backend && python manage.py migrate")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Mac M1 System Test - Customer Churn Prediction")
        print("=" * 60)
        
        self.test_python_environment()
        self.test_file_structure()
        self.test_django_server()
        self.test_streamlit_server()
        self.test_data_operations()
        self.test_ml_operations()
        self.test_analytics()
        
        self.print_summary()

def main():
    """Main function"""
    tester = MacM1SystemTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
