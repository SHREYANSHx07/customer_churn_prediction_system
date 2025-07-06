# Customer Churn Prediction System

A comprehensive customer churn prediction system built with Django REST API backend and Streamlit frontend, featuring machine learning model training, predictions, and analytics.

## 🚀 Features

### 📊 Dashboard
- Real-time customer statistics and churn metrics
- Visual charts showing customer distribution and churn rates
- Key performance indicators (KPIs)

### 📁 Data Management
- **Upload CSV**: Import customer data from CSV files
- **Generate Sample Data**: Create synthetic customer data for testing
- Data validation and preprocessing
- Customer data viewing and management

### 🤖 Model Training
- **Random Forest Algorithm**: Train churn prediction models
- **Mac M1 Optimization**: Optimized for Apple Silicon processors
- Model performance metrics and evaluation
- Feature importance analysis
- Model versioning and management

### 🔮 Predictions
- **Generate Predictions**: Predict churn for existing customers
- **View Predictions**: Browse and analyze prediction results
- Confidence scores and probability distributions
- Individual customer risk assessment

### 📈 Analytics & Insights
- Comprehensive churn analytics dashboard
- Customer segmentation analysis
- Trend analysis and historical comparisons
- Interactive visualizations with Plotly

### 💾 Export
- **Export Predictions**: Download customer predictions as CSV
- Data export for further analysis
- Formatted reports

## 🛠️ Technology Stack

### Backend (Django)
- **Django 4.2.7**: Web framework
- **Django REST Framework**: API development
- **Django CORS Headers**: Cross-origin resource sharing
- **SQLite**: Default database (easily configurable)

### Frontend (Streamlit)
- **Streamlit 1.28.1**: Web app framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Statistical plotting

### Machine Learning
- **Scikit-learn**: Machine learning algorithms
- **Random Forest**: Primary classification algorithm
- **Model persistence**: Joblib for model serialization

## 📋 Prerequisites

- Python 3.9+
- pip (Python package manager)
- Virtual environment (recommended)

## 🔧 Installation & Setup

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd customer-churn-django-streamlit

# Or download and extract the ZIP file
```

### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Django Backend
```bash
# Navigate to Django backend directory
cd django_backend

# Run database migrations
python manage.py migrate

# Create superuser (optional, for admin access)
python manage.py createsuperuser

# Return to project root
cd ..
```

## 🚀 Running the Application

### Option 1: Automatic Setup (Recommended)
Use the provided script to run both servers automatically:

```bash
python run_servers.py
```

This script will:
- Check if ports 8000 and 8501 are available
- Start the Django backend server on port 8000
- Start the Streamlit frontend on port 8501
- Monitor both processes
- Provide easy shutdown options

### Option 2: Manual Setup
If you prefer to run servers manually:

#### Start Django Backend
```bash
# Terminal 1
cd django_backend
python manage.py runserver 8000
```

#### Start Streamlit Frontend
```bash
# Terminal 2 (new terminal window)
streamlit run streamlit_app.py --server.port 8501
```

### 🌐 Access the Application

- **Streamlit Frontend**: http://localhost:8501
- **Django Admin**: http://localhost:8000/admin
- **API Endpoints**: http://localhost:8000/api/

## 📚 API Endpoints

### Customer Management
- `GET /api/customers/` - List all customers
- `POST /api/customers/` - Create new customer
- `GET /api/customers/{id}/` - Get customer details
- `PUT /api/customers/{id}/` - Update customer
- `DELETE /api/customers/{id}/` - Delete customer

### Data Management
- `POST /api/upload-csv/` - Upload customer data from CSV
- `POST /api/generate-sample-data/` - Generate sample customer data

### Machine Learning
- `POST /api/train-model/` - Train a new churn prediction model
- `GET /api/models/` - List all trained models
- `POST /api/predict/` - Generate churn predictions
- `GET /api/predictions/` - Retrieve predictions

### Analytics
- `GET /api/analytics/` - Get comprehensive analytics data
- `GET /api/customer-analytics/` - Get customer-specific analytics

### Export
- `GET /api/export-predictions/` - Export predictions as CSV

## 🗂️ Project Structure

```
customer-churn-django-streamlit/
├── django_backend/              # Django REST API
│   ├── churn_app/              # Main Django app
│   │   ├── models.py           # Data models
│   │   ├── views.py            # API views
│   │   ├── serializers.py      # API serializers
│   │   └── urls.py             # URL routing
│   ├── django_backend/         # Django settings
│   └── manage.py               # Django management
├── streamlit_app.py            # Streamlit frontend
├── run_servers.py              # Server management script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Configuration

### Django Settings
- Database: SQLite (default) - easily configurable in `django_backend/settings.py`
- Debug mode: Enabled for development
- CORS: Configured for Streamlit frontend

### Streamlit Configuration
- Default port: 8501
- Page configuration: Wide layout, custom page title
- API base URL: http://localhost:8000

## 🧪 Sample Data

The system can generate sample customer data with the following features:
- Customer demographics (age, gender, location)
- Service usage patterns
- Account information
- Billing history
- Churn labels for training

## 🤖 Machine Learning Model

### Features Used for Prediction
- Customer tenure
- Monthly charges
- Total charges
- Service subscriptions
- Contract type
- Payment method
- Demographics

### Model Performance
- Algorithm: Random Forest Classifier
- Cross-validation scoring
- Feature importance analysis
- Confusion matrix and classification reports

## 📊 Analytics Dashboard

The analytics section provides:
- **Churn Rate Analysis**: Overall and segmented churn rates
- **Customer Segmentation**: Behavioral and demographic segments
- **Revenue Impact**: Financial analysis of churn
- **Trend Analysis**: Historical churn patterns
- **Predictive Insights**: Future churn predictions

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Use `run_servers.py` script which automatically checks port availability
   - Or manually kill processes using the ports

2. **Import Errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt` again

3. **Database Issues**
   - Run `python manage.py migrate` in the django_backend directory
   - Delete `db.sqlite3` file and re-run migrations for a fresh start

4. **API Connection Issues**
   - Ensure Django backend is running on port 8000
   - Check CORS configuration in Django settings

### Getting Help

If you encounter issues:
1. Check that both servers are running
2. Verify all dependencies are installed
3. Ensure virtual environment is activated
4. Check the console output for error messages

## 🔄 Development

### Adding New Features
1. Backend: Add new API endpoints in `django_backend/churn_app/views.py`
2. Frontend: Add new pages in `streamlit_app.py`
3. Models: Update data models in `django_backend/churn_app/models.py`

### Testing
- Django: `python manage.py test` in django_backend directory
- Manual testing through Streamlit interface

## 📝 License

This project is available for educational and commercial use.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements.

---

**Happy Predicting! 🎯**
