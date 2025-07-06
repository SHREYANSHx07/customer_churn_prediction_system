"""
Models for Customer Churn Prediction
Optimized for Mac M1 with proper field handling
"""

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import json

class Customer(models.Model):
    # Customer identification
    customer_id = models.CharField(max_length=50, unique=True)
    
    # Demographics
    gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female')], default='Male')
    senior_citizen = models.BooleanField(default=False)
    partner = models.CharField(max_length=10, choices=[('Yes', 'Yes'), ('No', 'No')], default='No')
    dependents = models.CharField(max_length=10, choices=[('Yes', 'Yes'), ('No', 'No')], default='No')
    
    # Account information
    tenure = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(100)], default=0)
    
    # Services
    phone_service = models.CharField(max_length=10, choices=[('Yes', 'Yes'), ('No', 'No')], default='No')
    multiple_lines = models.CharField(max_length=20, choices=[
        ('Yes', 'Yes'), ('No', 'No'), ('No phone service', 'No phone service')
    ], default='No')
    
    internet_service = models.CharField(max_length=20, choices=[
        ('DSL', 'DSL'), ('Fiber optic', 'Fiber optic'), ('No', 'No')
    ], default='No')
    
    online_security = models.CharField(max_length=20, choices=[
        ('Yes', 'Yes'), ('No', 'No'), ('No internet service', 'No internet service')
    ], default='No')
    
    online_backup = models.CharField(max_length=20, choices=[
        ('Yes', 'Yes'), ('No', 'No'), ('No internet service', 'No internet service')
    ], default='No')
    
    device_protection = models.CharField(max_length=20, choices=[
        ('Yes', 'Yes'), ('No', 'No'), ('No internet service', 'No internet service')
    ], default='No')
    
    tech_support = models.CharField(max_length=20, choices=[
        ('Yes', 'Yes'), ('No', 'No'), ('No internet service', 'No internet service')
    ], default='No')
    
    streaming_tv = models.CharField(max_length=20, choices=[
        ('Yes', 'Yes'), ('No', 'No'), ('No internet service', 'No internet service')
    ], default='No')
    
    streaming_movies = models.CharField(max_length=20, choices=[
        ('Yes', 'Yes'), ('No', 'No'), ('No internet service', 'No internet service')
    ], default='No')
    
    # Contract and billing
    contract = models.CharField(max_length=20, choices=[
        ('Month-to-month', 'Month-to-month'),
        ('One year', 'One year'),
        ('Two year', 'Two year')
    ], default='Month-to-month')
    
    paperless_billing = models.CharField(max_length=10, choices=[('Yes', 'Yes'), ('No', 'No')], default='No')
    
    payment_method = models.CharField(max_length=30, choices=[
        ('Electronic check', 'Electronic check'),
        ('Mailed check', 'Mailed check'),
        ('Bank transfer (automatic)', 'Bank transfer (automatic)'),
        ('Credit card (automatic)', 'Credit card (automatic)')
    ], default='Electronic check')
    
    # Financial
    monthly_charges = models.FloatField(validators=[MinValueValidator(0)], default=0.0)
    total_charges = models.FloatField(validators=[MinValueValidator(0)], default=0.0)
    
    # Churn prediction
    churn = models.CharField(max_length=10, choices=[('Yes', 'Yes'), ('No', 'No')], null=True, blank=True)
    churn_probability = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])
    risk_level = models.CharField(max_length=10, choices=[
        ('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')
    ], null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'customers'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Customer {self.customer_id}"
    
    def to_dict(self):
        """Convert model instance to dictionary for ML processing"""
        return {
            'customer_id': self.customer_id,
            'gender': self.gender,
            'senior_citizen': 1 if self.senior_citizen else 0,
            'partner': self.partner,
            'dependents': self.dependents,
            'tenure': self.tenure,
            'phone_service': self.phone_service,
            'multiple_lines': self.multiple_lines,
            'internet_service': self.internet_service,
            'online_security': self.online_security,
            'online_backup': self.online_backup,
            'device_protection': self.device_protection,
            'tech_support': self.tech_support,
            'streaming_tv': self.streaming_tv,
            'streaming_movies': self.streaming_movies,
            'contract': self.contract,
            'paperless_billing': self.paperless_billing,
            'payment_method': self.payment_method,
            'monthly_charges': self.monthly_charges,
            'total_charges': self.total_charges,
            'churn': self.churn,
        }

class MLModel(models.Model):
    """Store ML model metadata and performance metrics"""
    name = models.CharField(max_length=100, default='churn_prediction_model')
    version = models.CharField(max_length=20, default='1.0')
    algorithm = models.CharField(max_length=50, default='RandomForest')
    
    # Performance metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    
    # Training metadata
    training_samples = models.IntegerField(default=0)
    feature_count = models.IntegerField(default=0)
    feature_names = models.TextField(default='[]')  # JSON string
    
    # Model file path (relative to project)
    model_file_path = models.CharField(max_length=255, null=True, blank=True)
    
    # Timestamps
    trained_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'ml_models'
        ordering = ['-trained_at']
    
    def __str__(self):
        return f"{self.name} v{self.version}"
    
    def get_feature_names(self):
        """Get feature names as list"""
        try:
            return json.loads(self.feature_names)
        except:
            return []
    
    def set_feature_names(self, names):
        """Set feature names from list"""
        self.feature_names = json.dumps(names)
