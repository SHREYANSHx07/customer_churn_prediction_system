"""
Admin configuration for Customer Churn Prediction
"""

from django.contrib import admin
from .models import Customer, MLModel

@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = [
        'customer_id', 'gender', 'tenure', 'monthly_charges', 
        'total_charges', 'churn', 'risk_level', 'created_at'
    ]
    list_filter = ['gender', 'churn', 'risk_level', 'contract', 'internet_service']
    search_fields = ['customer_id', 'gender']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('customer_id', 'gender', 'senior_citizen', 'partner', 'dependents')
        }),
        ('Account Details', {
            'fields': ('tenure', 'monthly_charges', 'total_charges')
        }),
        ('Services', {
            'fields': (
                'phone_service', 'multiple_lines', 'internet_service',
                'online_security', 'online_backup', 'device_protection',
                'tech_support', 'streaming_tv', 'streaming_movies'
            )
        }),
        ('Contract & Billing', {
            'fields': ('contract', 'paperless_billing', 'payment_method')
        }),
        ('Predictions', {
            'fields': ('churn', 'churn_probability', 'risk_level')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'version', 'algorithm', 'accuracy', 
        'training_samples', 'is_active', 'trained_at'
    ]
    list_filter = ['algorithm', 'is_active', 'trained_at']
    readonly_fields = ['trained_at']
    
    fieldsets = (
        ('Model Information', {
            'fields': ('name', 'version', 'algorithm', 'is_active')
        }),
        ('Performance Metrics', {
            'fields': ('accuracy', 'precision', 'recall', 'f1_score')
        }),
        ('Training Details', {
            'fields': ('training_samples', 'feature_count', 'feature_names')
        }),
        ('File Information', {
            'fields': ('model_file_path', 'trained_at')
        }),
    )
