
from django.urls import path

from . import views

urlpatterns = [

    # - Homepage

    path('', views.home, name=""),


]









