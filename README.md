# Aplicación de modelos de aprendizaje automático en microcontroladores

---
Este repositorio contiene el código con las implementaciones utilizadas para el Trabajo de Fin de Grado titulado
Aplicación de modelos de aprendizaje automático en microcontroladores.

Este repositorio no contiene los datos utilizados para entrenar los modelos de aprendizaje automático del projecto,
debido a limitaciones de alojamiento y para conservar la privacidad de las personas que han contribuido en la captura
de los mismos.

Algunos de los notebooks usados tienen scripts de python con el mismo nombre asociados que contienen funciones que se
utilizan en dichos notebooks.
## Captura de imágenes
Para la captura de imágenes destinadas al entrenamiento de los modelos se ha usado:
- [Arduino_cam_collection.pde](https://github.com/dainelli98/tfg-tinyml/blob/main/image%20capture/Arduino_cam_collection.pde):
  Script de Processing que recibe imágenes capturadas con Arduino, las etiqueta y las guarda.
- [Embedded_cam_collection.ino](https://github.com/dainelli98/tfg-tinyml/blob/main/image%20capture/Embedded_cam_collection/Embedded_cam_collection.ino):
  Programa para Arduino Nano 33 BLE Sense que captura imágenes y las transmite al equipo conectado al microcontrolador.
## Preprocesado
Procedimientos que se han empleado para el preprocesado de datos de imagen y audio junto con los scripts usados:
- [data_count.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/preprocessing/data_count.ipynb):
  Notebook que muestra los datos de los que se dispone inicialmente sin preprocesar.
- [data_count.py](https://github.com/dainelli98/tfg-tinyml/blob/main/preprocessing/data_count.py):
  Script de Python que contiene las funciones que realizan el contado de datos.
- [audio_preprocessing.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/preprocessing/audio_preprocessing.ipynb):
  Notebook que muestra las técnicas empleadas para el preprocesado de datos de audio.
- [image_preprocessing.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/preprocessing/image_preprocessing.ipynb):
  Notebook que muestra las técnicas empleadas para el preprocesado de datos de imagen.
- [image_preprocessing.py](https://github.com/dainelli98/tfg-tinyml/blob/main/preprocessing/image_preprocessing.py):
  Script de Python que contiene las funciones que realizan el preprocesado de los datos de imagen.
## Entrenamiento de modelos
Notebooks y scripts de Python complementarios para el entrenamiento de modelos de aprendizaje automático de sonido e
imagen.