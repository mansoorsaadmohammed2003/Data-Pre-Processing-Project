from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_file, name='upload_file'),
    path('download/', views.download_cleaned_data, name='download_cleaned_data'),
    path('train_model/', views.train_model, name='train_model'),
    path('download_model/<str:filename>/', views.download_model, name='download_model'),
]
