"""
Serializers for Customer Churn Prediction API
"""

from rest_framework import serializers
from .models import Customer, MLModel

class CustomerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Customer
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')
    
    def validate_monthly_charges(self, value):
        if value < 0:
            raise serializers.ValidationError("Monthly charges cannot be negative")
        return value
    
    def validate_total_charges(self, value):
        if value < 0:
            raise serializers.ValidationError("Total charges cannot be negative")
        return value
    
    def validate_tenure(self, value):
        if value < 0 or value > 100:
            raise serializers.ValidationError("Tenure must be between 0 and 100 months")
        return value

class CustomerCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating customers with flexible field mapping"""
    
    class Meta:
        model = Customer
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at', 'churn_probability', 'risk_level')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make most fields optional for flexible CSV import
        for field_name, field in self.fields.items():
            if field_name not in ['customer_id']:
                field.required = False

class MLModelSerializer(serializers.ModelSerializer):
    feature_names_list = serializers.SerializerMethodField()
    
    class Meta:
        model = MLModel
        fields = '__all__'
        read_only_fields = ('trained_at',)
    
    def get_feature_names_list(self, obj):
        return obj.get_feature_names()

class PredictionResultSerializer(serializers.Serializer):
    customer_id = serializers.CharField()
    churn_probability = serializers.FloatField()
    risk_level = serializers.CharField()
    prediction = serializers.CharField()

class BulkUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    
    def validate_file(self, value):
        if not value.name.endswith('.csv'):
            raise serializers.ValidationError("Only CSV files are allowed")
        
        # Check file size (10MB limit)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 10MB")
        
        return value

class URLUploadSerializer(serializers.Serializer):
    url = serializers.URLField()
    
    def validate_url(self, value):
        if not value.endswith('.csv'):
            raise serializers.ValidationError("URL must point to a CSV file")
        return value
