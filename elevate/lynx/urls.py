
from django.urls import path

from . import views

urlpatterns = [

    # - Homepage
    path('images/', views.images_view, name='images'),
    path('images/<image_name>/', views.image_detail, name='image_detail'),
    path('', views.home, name=""),


]









