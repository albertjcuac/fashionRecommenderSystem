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




CLASSES = 25

classDic = {
    0: 'CasualMen',
    1: 'CasualWomen',
    2: 'EthnicMen',
    3: 'CasualBoys',
    4: 'FormalMen',
    5: 'CasualGirls',
    6: 'CasualUnisex',
    7: 'SportsMen',
    8: 'EthnicWomen',
    9: 'SportsWomen',
    10: 'FormalWomen',
    11: 'SportsUnisex',
    12: 'SportsBoys',
    13: 'Smart CasualWomen',
    14: 'TravelUnisex',
    15: 'Smart CasualMen',
    16: 'PartyWomen',
    17: 'EthnicBoys',
    18: 'EthnicGirls',
    19: 'TravelWomen',
    20: 'HomeUnisex',
    21: 'PartyMen',
    22: 'SportsGirls',
    23: 'TravelMen',
    24: 'FormalUnisex'
} 
transformaciones = transforms.Compose([transforms.Resize((64,64)),
                                      transforms.ToTensor(), #0 - 255--->0 - 1
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #0 - 1--->-1 - 1
                                     ])

class CNNModel(pl.LightningModule):
    def __init__(self):
        #shape=[batchsize,canalesentrada,ancho,alto]
        #imagen entrada [64, 3, 64, 64])
        super().__init__()
    #EXTRACCIÓN DE CARACTERÍSTICAS
        #1 BLOQUE CONV
        self.cnv = nn.Conv2d(3,40,5,2)#[64, 40, 30, 30]
        self.rel = nn.ReLU()                    
        self.bn = nn.BatchNorm2d(40)  
        self.mxpool = nn.MaxPool2d(2)#[64, 40, 15, 15]
        #2 BLOQUE CONV
        self.cnv2 = nn.Conv2d(40,55,5,2)#[64, 55, 6, 6])
        self.rel2 = nn.ReLU()                    
        self.bn2 = nn.BatchNorm2d(55)  
        self.mxpool2 = nn.MaxPool2d(2)#[64, 55, 3, 3]
        
        
    #CARACTERIZACIÓN
        self.flat = nn.Flatten()#[64,55x3x3=495]
        self.fc1 = nn.Linear(495,495)
        self.fc2 = nn.Linear(495,300)
        self.fc3 = nn.Linear(300,CLASSES)
        self.softmax = nn.Softmax()
        self.accuracy = metrics.Accuracy(task='multiclass',num_classes=25)   #predicciones correctas/total de predicciones
                                                                   
        
    def forward(self,x):
       # print('antes conv',x.shape)
        out = self.cnv(x)
        #print('despues conv',out.shape)
        out = self.rel(out)
       # print('despues relu',out.shape)
        out = self.bn(out)
        #print('despues batchnorm',out.shape)
        out = self.mxpool(out)
        #rint('despues maxpool',out.shape)
        
        
        out = self.cnv2(out)
        #print('despues conv2',out.shape)
        out = self.rel2(out)
        #print('despues relu2',out.shape)
        out = self.bn2(out)
        #print('despues batchnorm2',out.shape)
        out = self.mxpool2(out)
       # print('despues maxpool2',out.shape)
        
        out = self.flat(out)
       # print('despues flat',out.shape)
        out = self.rel(self.fc1(out))
        out = self.rel(self.fc2(out))   
        out = self.fc3(out)
        return out

    def loss_fn(self,out,target):
        return nn.CrossEntropyLoss()(out.view(-1,CLASSES),target)
    
    def configure_optimizers(self):
        LR = 1e-3
        optimizer = torch.optim.AdamW(self.parameters(),lr=LR)
        return optimizer
    
    def predict(self, x):
        with torch.no_grad():
            out = self(x)
            out=nn.Softmax(-1)(out)
            return torch.argmax(out, axis=1)

    def training_step(self,batch,batch_idx):
        x,y = batch
        imgs = x.view(-1,3,64,64)
        labels = y.view(-1)
        out = self(imgs)
        loss = self.loss_fn(out,labels)
        out = nn.Softmax(-1)(out)
        logits = torch.argmax(out,dim=1)
        accu = self.accuracy(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accu, prog_bar=True)
        return loss       

    def validation_step(self,batch,batch_idx):
        x,y = batch
        imgs = x.view(-1,3,64,64)
        labels = y.view(-1)
        out = self(imgs)
        loss = self.loss_fn(out,labels)
        out = nn.Softmax(-1)(out) 
        logits = torch.argmax(out,dim=1)
        accu = self.accuracy(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accu, prog_bar=True)
        return loss, accu



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
            'link': '/images/1163.jpg',
        },
        {
            'url': '/static/1571.jpg',
            'name': 'Imagen 2',
            'link': '/images/1571.jpg',
        },
        {
            'url': '/static/24539.jpg',
            'name': 'Imagen 3',
            'link': '/images/24539.jpg',
        }, 
        {
            'url': '/static/37802.jpg',
            'name': 'Imagen 4',
            'link': '/images/37802.jpg',
        }, 
        {
            'url': '/static/59018.jpg',
            'name': 'Imagen 5',
            'link': '/images/59018.jpg',
        }, 
        {
            'url': '/static/10054.jpg',
            'name': 'Imagen 6',
            'link': '/images/10054.jpg',
        }, 
        
        
    ]

    return render(request, 'images.html', {'images': images})



# views.py

def image_detail(request, image_name):
    PATH="lynx/models/prueba6.pth"
   
    #Cargo mi modelo ya entrenado
    state = torch.load(PATH,map_location=torch.device('cpu'))

    new_model =CNNModel()
    new_model.load_state_dict(state)

    img = Image.open('lynx/static/'+image_name)
    t_img = transformaciones(img).unsqueeze(0)
    prediction = new_model.predict(t_img)

    etiqueta=classDic[prediction.item()]
    print('Valor predicho: ',etiqueta )














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
            'name': 'Imagen 3',
            'link': '/images/2',
        },
    ]
   
    return render(request, 'image_detail.html', {'images': images})

