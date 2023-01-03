from django.shortcuts import render
from numpy import loadtxt
import pandas as pd
from PIL import Image
from fastcore.all import *
from torch.utils.data import Dataset ,DataLoader
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import torch
from torch import nn
import torchmetrics as metrics
import pytorch_lightning as pl



# - Homepage

def home(request):

    return render(request, 'lynx/index.html')


from django.shortcuts import render

def images_view(request):
    # Aquí puedes obtener las imágenes que quieres mostrar en la página
    # desde la base de datos o cualquier otro lugar.
    # En este ejemplo, suponemos que tienes una lista de diccionarios
    # con información sobre las imágenes.
    images = [
        {
            'url': '/static/1163.jpg',
            'name': 'Imagen 1',
            'link': '/images/1',
        },
        {
            'url': '/static/1535.jpg',
            'name': 'Imagen 2',
            'link': '/images/2',
        },
        # ...
    ]

    return render(request, 'images.html', {'images': images})



# views.py

def image_detail(request, image_id):

    images = [
        {
            'url': '/static/1163.jpg',
            'name': 'Imagen 1',
            'link': '/images/1',
        },
        {
            'url': '/static/1535.jpg',
            'name': 'Imagen 2',
            'link': '/images/2',
        },
        {
            'url': '/static/1571.jpg',
            'name': 'Imagen 2',
            'link': '/images/2',
        },
    ]
   
    return render(request, 'image_detail.html', {'images': images})

