# Customer Churn Prediction - Deployment Guide

## Overview
This application consists of two components:
1. **Django REST API Backend** - Handles data processing, ML model predictions, and analytics
2. **Streamlit Frontend** - Provides the user interface for predictions and analytics

## Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- Streamlit Cloud account (https://streamlit.io/cloud)
- Django backend deployed and accessible via HTTPS

### Step 1: Deploy Django Backend
Since Streamlit Cloud only runs Streamlit apps, deploy your Django backend separately:

**Recommended platforms:**
- **Heroku** (free tier available)
- **Railway** (simple deployment)
- **PythonAnywhere** (Django-friendly)
- **DigitalOcean App Platform**
- **AWS/GCP/Azure** (for production)

### Step 2: Update API Configuration
Before deploying to Streamlit Cloud, update the API endpoint in `streamlit_app.py`:

```python
# Change this line:
API_BASE_URL = "http://127.0.0.1:8000/api"

# To your deployed Django backend URL:
API_BASE_URL = "https://your-django-backend-url/api"
```

### Step 3: Configure Django Backend for Production
Ensure your Django backend has proper CORS settings in `settings.py`:

```python
# Add your Streamlit Cloud URL to CORS_ALLOWED_ORIGINS
CORS_ALLOWED_ORIGINS = [
    "https://your-streamlit-app.streamlit.app",
    "http://localhost:3000",  # for local development
]

# Or for development (less secure):
CORS_ALLOW_ALL_ORIGINS = True
```

### Step 4: Prepare for Streamlit Cloud
1. **Create a GitHub repository** with your project files
2. **Use the streamlit-specific requirements file:**
   - Rename `requirements-streamlit.txt` to `requirements.txt` for Streamlit Cloud
   - Or create a `requirements.txt` with only frontend dependencies

### Step 5: Deploy to Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set main file path to `streamlit_app.py`
6. Click "Deploy"

### Step 6: Environment Variables (if needed)
If your app requires environment variables:
1. In Streamlit Cloud dashboard, go to your app settings
2. Add environment variables in the "Secrets" section
3. Use `st.secrets` in your code to access them

## Local Development

### Running Both Servers Locally
Use the provided script to run both Django and Streamlit simultaneously:

```bash
python run_servers.py
```

This will start:
- Django backend on `http://127.0.0.1:8000`
- Streamlit frontend on `http://localhost:8501`

### Manual Setup
1. **Start Django backend:**
   ```bash
   cd django_backend
   python manage.py runserver
   ```

2. **Start Streamlit frontend (in another terminal):**
   ```bash
   streamlit run streamlit_app.py
   ```

## Troubleshooting

### Common Issues:
1. **CORS errors**: Ensure your Django backend allows requests from your Streamlit Cloud URL
2. **API connection issues**: Verify your Django backend is accessible and the API_BASE_URL is correct
3. **Package conflicts**: Use the provided `requirements-streamlit.txt` for Streamlit Cloud
4. **Model loading errors**: Ensure your ML models are properly saved and accessible

### Debugging Tips:
- Check Streamlit Cloud logs for error messages
- Test your Django API endpoints directly using tools like Postman
- Use `st.write()` to debug variable values in your Streamlit app

## File Structure for Deployment
```
customer-churn-django-streamlit/
├── streamlit_app.py          # Main Streamlit app
├── requirements.txt          # Streamlit Cloud requirements
├── DEPLOYMENT_GUIDE.md       # This file
├── django_backend/           # Django backend (deploy separately)
│   ├── manage.py
│   ├── requirements.txt      # Django requirements
│   └── ...
└── ...
```

## Production Considerations
- Use environment variables for sensitive configuration
- Enable HTTPS for your Django backend
- Consider using a CDN for static assets
- Implement proper logging and monitoring
- Use a production-grade database (PostgreSQL, MySQL)
- Set up automated backups for your data
