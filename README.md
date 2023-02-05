# Sistema de Recomendación para Artículos de Moda

Se trata de una aplicación simple cuyo objetivo es el de mostar de manera visual y sencilla el funcionamiento de un sistema de recomendación con un filtrado basado en el contenido, desarrollado por Alberto Monedero Martín como parte de su TFG. Este sistema consta de 2 bloques:
* Bloque 1 - Clasificación del producto de entrada mediante CNN para la obtención de candidatos a recomendar.
* Bloque 2 - Obtención de los N candidatos más parecidos a la imagen de entrada basándonos en la similitud coseno.

 Para entender en profundidad: [Notebook Kaggle](https://www.kaggle.com/code/albertomonedero/fashioncnn) 

## Despliegue

Con las siguientes instrucciones se explica como desplegar en local la aplicación.
### prerrequisitos
[Python 3.9.8]https://www.python.org/downloads/release/python-398/
Puede que sea compatible con otras versiones de Python, teniendo en cuenta que Pytorch solo da soporte a las versiones 3.7-3.9.
### Crear entorno virtual
py -m venv env
### Activar el entorno virtual
env\Scripts\activate
### Instalar los requisitos 
pip install -r requirements.txt
### Lanzar aplicación en local (desde la carpeta "elevate")
py manage.py runserver

## Tecnologías

* [Django](https://www.djangoproject.com/) - Framework utilizado
* [Pytorch](https://pytorch.org/) - librería aprendizaje automático
* Python, html, css y Bootstrap 4


## Autor

* **Alberto Monedero Martín** - [github](https://github.com/albertjcuac)
