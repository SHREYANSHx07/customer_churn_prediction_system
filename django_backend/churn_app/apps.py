"""
App configuration for churn_app
"""

from django.apps import AppConfig

class ChurnAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'churn_app'
    verbose_name = 'Customer Churn Prediction'
