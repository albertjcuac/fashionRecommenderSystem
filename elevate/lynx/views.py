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
from django.shortcuts import redirect


ds=pd.read_csv('lynx/csv/custom.csv')

CLASSES = 25

classDic = {
    0: 'Casual Men',
    1: 'Casual Women',
    2: 'Ethnic Men',
    3: 'Casual Boys',
    4: 'Formal Men',
    5: 'Casual Girls',
    6: 'Casual Unisex',
    7: 'Sports Men',
    8: 'Ethnic Women',
    9: 'Sports Women',
    10: 'Formal Women',
    11: 'Sports Unisex',
    12: 'Sports Boys',
    13: 'Smart Casual Women',
    14: 'Travel Unisex',
    15: 'Smart Casual Men',
    16: 'Party Women',
    17: 'Ethnic Boys',
    18: 'Ethnic Girls',
    19: 'Travel Women',
    20: 'Home Unisex',
    21: 'Party Men',
    22: 'Sports Girls',
    23: 'Travel Men',
    24: 'Formal Unisex'
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

from img2vec_pytorch import Img2Vec
cos = nn.CosineSimilarity(eps=1e-6)
img2vec = Img2Vec(cuda=False, model='resnet-18')
def N_mas_parecidas(imagen,n,imagenes):
    
    vec1 = img2vec.get_vec(imagen, tensor=True).reshape(512) #representación vectorial imagen de entrada
    dic={}#diccionario con imagenes y similitudes
    similitudes=[]
    recomendadas={}
    for im in imagenes: #cargo el diccionario con cada imagen como clave y la similitud como valor
                        #y obtengo una lista con las n mayores similitudes, ordenada de mayor a menor
        candidata = Image.open('lynx'+im.get('url'))       
        vec2 = img2vec.get_vec(candidata.convert('RGB'), tensor=True).reshape(512)
        cos_sim = cos(vec1.unsqueeze(0),vec2.unsqueeze(0))
        if(cos_sim<1): #no quiero devolver la propia imagen cono recomendación
            dic[im.get('name')]=cos_sim
            similitudes.append(cos_sim)
    n_mayor_similitud=sorted(list(set(similitudes)), reverse=True)[:n]
    for im in imagenes:
        if(dic.get(im.get('name')) in n_mayor_similitud):
            recomendadas[im.get('name')]=dic.get(im.get('name'))
            
    return recomendadas 


# - Homepage

def home(request):

    return render(request, 'lynx/index.html')
def contacto(request):

    return render(request, 'contacto.html')


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
            'url': '/static/3818.jpg',
            'name': 'Imagen 6',
            'link': '/images/3818.jpg',
        }, 
        
        
    ]

    return render(request, 'images.html', {'images': images})


from django import forms

class IntegerForm(forms.Form):
    integer_field = forms.IntegerField(label='¿Cuántas recomendaciones quieres?')

    def __init__(self, *args, **kwargs):
        self.img = kwargs.pop('img')
        self.images = kwargs.pop('images')
        super(IntegerForm, self).__init__(*args, **kwargs)



def image_detail(request, image_name):
    PATH="lynx/models/prueba6.pth"
   
    #Cargo mi modelo ya entrenado
    state = torch.load(PATH,map_location=torch.device('cpu'))

    new_model =CNNModel()
    new_model.load_state_dict(state)
    #Obtengo la predicción de su clase
    img = Image.open('lynx/static/'+image_name)
    t_img = transformaciones(img).unsqueeze(0)
    prediction = new_model.predict(t_img)

    etiqueta=classDic[prediction.item()]
    print('Valor predicho: ',etiqueta )
    #Obtengo las imágenes que pertenecen a la misma clase
    condicion_etiqueta = ds['target'] == prediction.item()
    candidatas=ds[condicion_etiqueta]
    print(candidatas)
    imagenes=candidatas.image.unique()

    images = []
    for imagen in imagenes:
        image = {
            'url': f'/static/images/{imagen}',
            'name': f'{imagen}',
            'link': f'/images/{imagen}',
        }
        images.append(image)
  
    form = IntegerForm(img=img, images=images)

    if request.method == 'POST':
        form = IntegerForm(request.POST, img=img, images=images)
        if form.is_valid():
            integer = form.cleaned_data['integer_field']
            recomendadas = N_mas_parecidas(img, integer, images)
            similares = []
            for image, similarity in recomendadas.items() :
                similar = {
                    'url': f'/static/images/{image}',
                    'name': f'{image}',
                    'similarity':f'{similarity}',
                    
                }
                img_url=f'/static/images/{image_name}'
                similares.append(similar)
            return render(request, 'similar_images.html', {'similares': similares,'img':img_url})
            

    return render(request, 'image_detail.html', {'form': form, 'images': images, 'etiqueta': etiqueta})





    






