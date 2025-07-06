"""
Views for Customer Churn Prediction API
Optimized for Mac M1 with robust error handling
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
import requests
from io import StringIO
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.db import transaction, models
from django.core.paginator import Paginator
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from rest_framework import status, viewsets
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FileUploadParser, JSONParser

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from .models import Customer, MLModel
from .serializers import (
    CustomerSerializer, CustomerCreateSerializer, MLModelSerializer,
    PredictionResultSerializer, BulkUploadSerializer, URLUploadSerializer
)

class CustomerViewSet(viewsets.ModelViewSet):
    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer
    
    def get_serializer_class(self):
        if self.action == 'create':
            return CustomerCreateSerializer
        return CustomerSerializer
    
    def list(self, request):
        """List customers with pagination and filtering"""
        queryset = self.get_queryset()
        
        # Apply filters
        risk_level = request.query_params.get('risk_level')
        if risk_level:
            queryset = queryset.filter(risk_level=risk_level)
        
        churn = request.query_params.get('churn')
        if churn:
            queryset = queryset.filter(churn=churn)
        
        # Pagination
        page_size = int(request.query_params.get('page_size', 50))
        page_number = int(request.query_params.get('page', 1))
        
        paginator = Paginator(queryset, page_size)
        page_obj = paginator.get_page(page_number)
        
        serializer = self.get_serializer(page_obj, many=True)
        
        return Response({
            'customers': serializer.data,
            'pagination': {
                'current_page': page_number,
                'total_pages': paginator.num_pages,
                'total_count': paginator.count,
                'page_size': page_size,
                'has_next': page_obj.has_next(),
                'has_previous': page_obj.has_previous(),
            }
        })

def clean_and_convert_data(df):
    """Clean and convert DataFrame for Mac M1 compatibility"""
    
    # Column mapping for different CSV formats
    column_mapping = {
        'customerID': 'customer_id',
        'Customer ID': 'customer_id',
        'ID': 'customer_id',
        'Gender': 'gender',
        'SeniorCitizen': 'senior_citizen',
        'Senior Citizen': 'senior_citizen',
        'Partner': 'partner',
        'Dependents': 'dependents',
        'Tenure': 'tenure',
        'tenure': 'tenure',
        'PhoneService': 'phone_service',
        'Phone Service': 'phone_service',
        'MultipleLines': 'multiple_lines',
        'Multiple Lines': 'multiple_lines',
        'InternetService': 'internet_service',
        'Internet Service': 'internet_service',
        'OnlineSecurity': 'online_security',
        'Online Security': 'online_security',
        'OnlineBackup': 'online_backup',
        'Online Backup': 'online_backup',
        'DeviceProtection': 'device_protection',
        'Device Protection': 'device_protection',
        'TechSupport': 'tech_support',
        'Tech Support': 'tech_support',
        'StreamingTV': 'streaming_tv',
        'Streaming TV': 'streaming_tv',
        'StreamingMovies': 'streaming_movies',
        'Streaming Movies': 'streaming_movies',
        'Contract': 'contract',
        'PaperlessBilling': 'paperless_billing',
        'Paperless Billing': 'paperless_billing',
        'PaymentMethod': 'payment_method',
        'Payment Method': 'payment_method',
        'MonthlyCharges': 'monthly_charges',
        'Monthly Charges': 'monthly_charges',
        'TotalCharges': 'total_charges',
        'Total Charges': 'total_charges',
        'Total Spend': 'total_charges',  # Alternative naming
        'Churn': 'churn',
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Ensure customer_id exists
    if 'customer_id' not in df.columns:
        if 'customerID' in df.columns:
            df['customer_id'] = df['customerID']
        else:
            df['customer_id'] = [f'CUST_{i:06d}' for i in range(len(df))]
    
    # Clean and convert data types
    for col in df.columns:
        if col in ['monthly_charges', 'total_charges']:
            # Handle numeric columns
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
        
        elif col == 'tenure':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype(int)
        
        elif col == 'senior_citizen':
            # Convert to boolean
            if df[col].dtype == 'object':
                df[col] = df[col].map({'Yes': True, 'No': False, 1: True, 0: False})
            else:
                df[col] = df[col].astype(bool)
        
        else:
            # String columns
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', 'No')
    
    # Set default values for missing columns
    default_values = {
        'gender': 'Male',
        'senior_citizen': False,
        'partner': 'No',
        'dependents': 'No',
        'tenure': 0,
        'phone_service': 'No',
        'multiple_lines': 'No',
        'internet_service': 'No',
        'online_security': 'No',
        'online_backup': 'No',
        'device_protection': 'No',
        'tech_support': 'No',
        'streaming_tv': 'No',
        'streaming_movies': 'No',
        'contract': 'Month-to-month',
        'paperless_billing': 'No',
        'payment_method': 'Electronic check',
        'monthly_charges': 0.0,
        'total_charges': 0.0,
    }
    
    for col, default_val in default_values.items():
        if col not in df.columns:
            df[col] = default_val
    
    return df

@api_view(['POST'])
def upload_customer_data(request):
    """Upload customer data from CSV file"""
    try:
        serializer = BulkUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        csv_file = serializer.validated_data['file']
        
        # Read CSV
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            return Response({
                'error': f'Failed to read CSV file: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if df.empty:
            return Response({
                'error': 'CSV file is empty'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Clean and convert data
        df = clean_and_convert_data(df)
        
        # Create customers
        created_count = 0
        errors = []
        
        with transaction.atomic():
            for index, row in df.iterrows():
                try:
                    customer_data = {
                        'customer_id': row.get('customer_id', f'CUST_{index:06d}'),
                        'gender': row.get('gender', 'Male'),
                        'senior_citizen': bool(row.get('senior_citizen', False)),
                        'partner': row.get('partner', 'No'),
                        'dependents': row.get('dependents', 'No'),
                        'tenure': int(row.get('tenure', 0)),
                        'phone_service': row.get('phone_service', 'No'),
                        'multiple_lines': row.get('multiple_lines', 'No'),
                        'internet_service': row.get('internet_service', 'No'),
                        'online_security': row.get('online_security', 'No'),
                        'online_backup': row.get('online_backup', 'No'),
                        'device_protection': row.get('device_protection', 'No'),
                        'tech_support': row.get('tech_support', 'No'),
                        'streaming_tv': row.get('streaming_tv', 'No'),
                        'streaming_movies': row.get('streaming_movies', 'No'),
                        'contract': row.get('contract', 'Month-to-month'),
                        'paperless_billing': row.get('paperless_billing', 'No'),
                        'payment_method': row.get('payment_method', 'Electronic check'),
                        'monthly_charges': float(row.get('monthly_charges', 0)),
                        'total_charges': float(row.get('total_charges', 0)),
                        'churn': row.get('churn', None),
                    }
                    
                    customer, created = Customer.objects.get_or_create(
                        customer_id=customer_data['customer_id'],
                        defaults=customer_data
                    )
                    
                    if created:
                        created_count += 1
                    else:
                        # Update existing customer
                        for key, value in customer_data.items():
                            setattr(customer, key, value)
                        customer.save()
                        created_count += 1
                
                except Exception as e:
                    errors.append(f'Row {index + 1}: {str(e)}')
        
        return Response({
            'message': f'Successfully processed {created_count} customers',
            'created_count': created_count,
            'total_rows': len(df),
            'errors': errors[:10]  # Limit errors shown
        })
    
    except Exception as e:
        return Response({
            'error': f'Upload failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def upload_from_url(request):
    """Upload customer data from URL"""
    try:
        serializer = URLUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        url = serializer.validated_data['url']
        
        # Download CSV from URL
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            csv_content = response.text
        except Exception as e:
            return Response({
                'error': f'Failed to download CSV from URL: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Read CSV from string
        try:
            df = pd.read_csv(StringIO(csv_content))
        except Exception as e:
            return Response({
                'error': f'Failed to parse CSV content: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if df.empty:
            return Response({
                'error': 'CSV file is empty'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Clean and convert data
        df = clean_and_convert_data(df)
        
        # Create customers
        created_count = 0
        updated_count = 0
        errors = []
        
        with transaction.atomic():
            for index, row in df.iterrows():
                try:
                    customer_data = {
                        'customer_id': row.get('customer_id', f'CUST_{index:06d}'),
                        'gender': row.get('gender', 'Male'),
                        'senior_citizen': bool(row.get('senior_citizen', False)),
                        'partner': row.get('partner', 'No'),
                        'dependents': row.get('dependents', 'No'),
                        'tenure': int(row.get('tenure', 0)),
                        'phone_service': row.get('phone_service', 'No'),
                        'multiple_lines': row.get('multiple_lines', 'No'),
                        'internet_service': row.get('internet_service', 'No'),
                        'online_security': row.get('online_security', 'No'),
                        'online_backup': row.get('online_backup', 'No'),
                        'device_protection': row.get('device_protection', 'No'),
                        'tech_support': row.get('tech_support', 'No'),
                        'streaming_tv': row.get('streaming_tv', 'No'),
                        'streaming_movies': row.get('streaming_movies', 'No'),
                        'contract': row.get('contract', 'Month-to-month'),
                        'paperless_billing': row.get('paperless_billing', 'No'),
                        'payment_method': row.get('payment_method', 'Electronic check'),
                        'monthly_charges': float(row.get('monthly_charges', 0)),
                        'total_charges': float(row.get('total_charges', 0)),
                        'churn': row.get('churn', None),
                    }
                    
                    customer, created = Customer.objects.get_or_create(
                        customer_id=customer_data['customer_id'],
                        defaults=customer_data
                    )
                    
                    if created:
                        created_count += 1
                    else:
                        # Update existing customer
                        for key, value in customer_data.items():
                            setattr(customer, key, value)
                        customer.save()
                        updated_count += 1
                
                except Exception as e:
                    errors.append(f'Row {index + 1}: {str(e)}')
        
        return Response({
            'message': f'Successfully processed {created_count + updated_count} customers',
            'created_count': created_count,
            'updated_count': updated_count,
            'total_rows': len(df),
            'errors': errors[:10]  # Limit errors shown
        })
    
    except Exception as e:
        return Response({
            'error': f'Upload failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def generate_sample_data(request):
    """Generate sample customer data for testing"""
    try:
        n_customers = request.data.get('n_customers', 100)
        n_customers = min(max(int(n_customers), 10), 1000)  # Limit between 10-1000
        
        # Generate sample data
        np.random.seed(42)  # For reproducible results
        
        customers_data = []
        for i in range(n_customers):
            customer_data = {
                'customer_id': f'SAMPLE_{i:06d}',
                'gender': np.random.choice(['Male', 'Female']),
                'senior_citizen': np.random.choice([True, False], p=[0.2, 0.8]),
                'partner': np.random.choice(['Yes', 'No'], p=[0.5, 0.5]),
                'dependents': np.random.choice(['Yes', 'No'], p=[0.3, 0.7]),
                'tenure': np.random.randint(0, 73),
                'phone_service': np.random.choice(['Yes', 'No'], p=[0.9, 0.1]),
                'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], p=[0.4, 0.5, 0.1]),
                'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], p=[0.3, 0.4, 0.3]),
                'online_security': np.random.choice(['Yes', 'No', 'No internet service'], p=[0.3, 0.4, 0.3]),
                'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], p=[0.3, 0.4, 0.3]),
                'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], p=[0.3, 0.4, 0.3]),
                'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], p=[0.3, 0.4, 0.3]),
                'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], p=[0.3, 0.4, 0.3]),
                'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], p=[0.3, 0.4, 0.3]),
                'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.5, 0.3, 0.2]),
                'paperless_billing': np.random.choice(['Yes', 'No'], p=[0.6, 0.4]),
                'payment_method': np.random.choice([
                    'Electronic check', 'Mailed check', 
                    'Bank transfer (automatic)', 'Credit card (automatic)'
                ], p=[0.3, 0.2, 0.25, 0.25]),
                'monthly_charges': round(np.random.uniform(18.0, 120.0), 2),
                'churn': np.random.choice(['Yes', 'No'], p=[0.27, 0.73]),  # Realistic churn rate
            }
            
            # Calculate total charges based on tenure and monthly charges
            customer_data['total_charges'] = round(
                customer_data['monthly_charges'] * customer_data['tenure'], 2
            )
            
            customers_data.append(customer_data)
        
        # Create customers in database
        created_count = 0
        with transaction.atomic():
            for customer_data in customers_data:
                customer, created = Customer.objects.get_or_create(
                    customer_id=customer_data['customer_id'],
                    defaults=customer_data
                )
                if created:
                    created_count += 1
        
        return Response({
            'message': f'Successfully generated {created_count} sample customers',
            'created_count': created_count,
            'total_requested': n_customers
        })
    
    except Exception as e:
        return Response({
            'error': f'Sample data generation failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def prepare_ml_data(customers_queryset):
    """Prepare customer data for ML model training - Mac M1 optimized"""
    
    # Convert to DataFrame
    data = []
    for customer in customers_queryset:
        data.append(customer.to_dict())
    
    df = pd.DataFrame(data)
    
    if df.empty:
        raise ValueError("No customer data available for training")
    
    # Feature engineering
    categorical_features = [
        'gender', 'partner', 'dependents', 'phone_service', 'multiple_lines',
        'internet_service', 'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies', 'contract',
        'paperless_billing', 'payment_method'
    ]
    
    numerical_features = ['senior_citizen', 'tenure', 'monthly_charges', 'total_charges']
    
    # Prepare features
    X = df[categorical_features + numerical_features].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature].astype(str))
        label_encoders[feature] = le
    
    # Handle target variable
    if 'churn' in df.columns and df['churn'].notna().any():
        y = df['churn'].fillna('No')  # Fill NaN with 'No'
        y = LabelEncoder().fit_transform(y)
    else:
        # If no churn data, create dummy target for training
        y = np.random.choice([0, 1], size=len(df), p=[0.73, 0.27])
    
    return X, y, label_encoders, X.columns.tolist()

@api_view(['POST'])
def train_model(request):
    """Train ML model for churn prediction - Mac M1 optimized"""
    try:
        # Get customer data
        customers = Customer.objects.all()
        
        if customers.count() < 10:
            return Response({
                'error': 'Need at least 10 customers to train model'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Prepare data
        try:
            X, y, label_encoders, feature_names = prepare_ml_data(customers)
        except Exception as e:
            return Response({
                'error': f'Data preparation failed: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model - Mac M1 optimized RandomForest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=1,  # Single job for Mac M1 stability
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        }
        
        # Save model
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        
        model_data = {
            'model': model,
            'label_encoders': label_encoders,
            'feature_names': feature_names
        }
        
        model_path = model_dir / 'churn_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save model metadata
        ml_model = MLModel.objects.create(
            name='churn_prediction_model',
            version='1.0',
            algorithm='RandomForest',
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            training_samples=len(X_train),
            feature_count=len(feature_names),
            model_file_path=str(model_path),
            is_active=True
        )
        
        ml_model.set_feature_names(feature_names)
        ml_model.save()
        
        # Deactivate old models
        MLModel.objects.filter(is_active=True).exclude(id=ml_model.id).update(is_active=False)
        
        return Response({
            'message': 'Model trained successfully',
            'model_id': ml_model.id,
            'metrics': metrics,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(feature_names),
            'feature_names': feature_names
        })
    
    except Exception as e:
        return Response({
            'error': f'Model training failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def predict_churn(request):
    """Generate churn predictions for all customers"""
    try:
        # Load model
        try:
            model_path = Path('models/churn_model.pkl')
            if not model_path.exists():
                return Response({
                    'error': 'No trained model found. Please train a model first.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            label_encoders = model_data['label_encoders']
            feature_names = model_data['feature_names']
            
        except Exception as e:
            return Response({
                'error': f'Failed to load model: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Get customers
        customers = Customer.objects.all()
        
        if customers.count() == 0:
            return Response({
                'error': 'No customers found'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Prepare data for prediction
        data = []
        for customer in customers:
            data.append(customer.to_dict())
        
        df = pd.DataFrame(data)
        
        # Prepare features (same as training)
        categorical_features = [
            'gender', 'partner', 'dependents', 'phone_service', 'multiple_lines',
            'internet_service', 'online_security', 'online_backup', 'device_protection',
            'tech_support', 'streaming_tv', 'streaming_movies', 'contract',
            'paperless_billing', 'payment_method'
        ]
        
        numerical_features = ['senior_citizen', 'tenure', 'monthly_charges', 'total_charges']
        
        X = df[categorical_features + numerical_features].copy()
        
        # Encode categorical variables using saved encoders
        for feature in categorical_features:
            if feature in label_encoders:
                le = label_encoders[feature]
                # Handle unseen categories
                X[feature] = X[feature].astype(str)
                X[feature] = X[feature].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Update customers with predictions
        updated_count = 0
        high_risk_count = 0
        
        with transaction.atomic():
            for i, customer in enumerate(customers):
                churn_prob = float(probabilities[i])
                prediction = 'Yes' if predictions[i] == 1 else 'No'
                
                # Determine risk level
                if churn_prob >= 0.7:
                    risk_level = 'High'
                    high_risk_count += 1
                elif churn_prob >= 0.4:
                    risk_level = 'Medium'
                else:
                    risk_level = 'Low'
                
                # Update customer
                customer.churn_probability = churn_prob
                customer.risk_level = risk_level
                customer.churn = prediction
                customer.save()
                
                updated_count += 1
        
        return Response({
            'message': 'Predictions generated successfully',
            'total_predictions': updated_count,
            'high_risk_count': high_risk_count,
            'medium_risk_count': updated_count - high_risk_count - sum(1 for c in customers if c.risk_level == 'Low'),
            'low_risk_count': sum(1 for c in customers if c.risk_level == 'Low')
        })
    
    except Exception as e:
        return Response({
            'error': f'Prediction failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_analytics(request):
    """Get analytics and insights about customer churn"""
    try:
        customers = Customer.objects.all()
        total_customers = customers.count()
        
        if total_customers == 0:
            return Response({
                'error': 'No customer data available'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Basic statistics
        analytics = {
            'total_customers': total_customers,
            'churn_rate': 0,
            'risk_distribution': {
                'High': customers.filter(risk_level='High').count(),
                'Medium': customers.filter(risk_level='Medium').count(),
                'Low': customers.filter(risk_level='Low').count(),
            },
            'demographics': {
                'gender': {
                    'Male': customers.filter(gender='Male').count(),
                    'Female': customers.filter(gender='Female').count(),
                },
                'senior_citizens': customers.filter(senior_citizen=True).count(),
                'with_partners': customers.filter(partner='Yes').count(),
                'with_dependents': customers.filter(dependents='Yes').count(),
            },
            'services': {
                'phone_service': customers.filter(phone_service='Yes').count(),
                'internet_service': {
                    'DSL': customers.filter(internet_service='DSL').count(),
                    'Fiber optic': customers.filter(internet_service='Fiber optic').count(),
                    'No': customers.filter(internet_service='No').count(),
                },
                'streaming_tv': customers.filter(streaming_tv='Yes').count(),
                'streaming_movies': customers.filter(streaming_movies='Yes').count(),
            },
            'contract_types': {
                'Month-to-month': customers.filter(contract='Month-to-month').count(),
                'One year': customers.filter(contract='One year').count(),
                'Two year': customers.filter(contract='Two year').count(),
            },
            'financial': {
                'avg_monthly_charges': customers.aggregate(avg=models.Avg('monthly_charges'))['avg'] or 0,
                'avg_total_charges': customers.aggregate(avg=models.Avg('total_charges'))['avg'] or 0,
                'avg_tenure': customers.aggregate(avg=models.Avg('tenure'))['avg'] or 0,
            }
        }
        
        # Calculate churn rate if churn data exists
        churned_customers = customers.filter(churn='Yes').count()
        if churned_customers > 0:
            analytics['churn_rate'] = round((churned_customers / total_customers) * 100, 2)
        
        return Response(analytics)
    
    except Exception as e:
        return Response({
            'error': f'Analytics generation failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def export_predictions(request):
    """Export customer predictions as CSV"""
    try:
        customers = Customer.objects.all()
        
        if customers.count() == 0:
            return Response({
                'error': 'No customer data available'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Prepare data for export
        data = []
        for customer in customers:
            data.append({
                'customer_id': customer.customer_id,
                'gender': customer.gender,
                'senior_citizen': customer.senior_citizen,
                'partner': customer.partner,
                'dependents': customer.dependents,
                'tenure': customer.tenure,
                'monthly_charges': customer.monthly_charges,
                'total_charges': customer.total_charges,
                'contract': customer.contract,
                'churn_probability': customer.churn_probability,
                'risk_level': customer.risk_level,
                'predicted_churn': customer.churn,
            })
        
        df = pd.DataFrame(data)
        
        # Convert to CSV
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        return Response({
            'csv_data': csv_content,
            'total_records': len(data),
            'filename': 'customer_churn_predictions.csv'
        })
    
    except Exception as e:
        return Response({
            'error': f'Export failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
