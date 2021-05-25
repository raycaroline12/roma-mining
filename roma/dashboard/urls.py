from django.urls import path
from . import dash_app
from . import views

urlpatterns = [
    path('dashboard', views.dashboard, name='dashboard'),
]