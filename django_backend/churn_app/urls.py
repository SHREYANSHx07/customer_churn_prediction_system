"""
URL configuration for churn_app
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'customers', views.CustomerViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('upload-csv/', views.upload_customer_data, name='upload_csv'),
    path('upload-from-url/', views.upload_from_url, name='upload_from_url'),
    path('generate-sample-data/', views.generate_sample_data, name='generate_sample_data'),
    path('train-model/', views.train_model, name='train_model'),
    path('predict/', views.predict_churn, name='predict_churn'),
    path('analytics/', views.get_analytics, name='get_analytics'),
    path('export/', views.export_predictions, name='export_predictions'),
]
