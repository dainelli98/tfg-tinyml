# Aplicación de modelos de aprendizaje automático en microcontroladores

---
Este repositorio contiene el código con las implementaciones utilizadas para el Trabajo de Fin de Grado titulado
Aplicación de modelos de aprendizaje automático en microcontroladores.

Este repositorio no contiene los datos utilizados para entrenar los modelos de aprendizaje automático del proyecto,
debido a limitaciones de alojamiento y para conservar la privacidad de las personas que han contribuido en la captura
de los mismos.

Algunos de los notebooks usados tienen scripts y módulos de Python con el mismo nombre asociados que contienen funciones
que se utilizan en dichos notebooks.

GitHub puede fallar al mostrar el contenido de algunos notebooks, para solucionar el problema puede ser necesario
recargar la página varias veces.
## Captura de imágenes
Para la captura de imágenes destinadas al entrenamiento de los modelos se ha usado:
- [Arduino_cam_collection.pde](https://github.com/dainelli98/tfg-tinyml/blob/main/image%20capture/Arduino_cam_collection/Arduino_cam_collection.pde):
  Script de Processing que recibe imágenes capturadas con Arduino, las etiqueta y las guarda.
- [Embedded_cam_collection.ino](https://github.com/dainelli98/tfg-tinyml/blob/main/image%20capture/Embedded_cam_collection/Embedded_cam_collection.ino):
  Programa para Arduino Nano 33 BLE Sense que captura imágenes y las transmite al equipo conectado al microcontrolador.
## Preprocesado
Procedimientos que se han empleado para el preprocesado de datos de imagen y audio junto con los scripts usados:
- [data_count.py](https://github.com/dainelli98/tfg-tinyml/blob/main/preprocessing/data_count.py): Módulo de Python que
  contiene las funciones que realizan el contado de datos.
- [data_count.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/preprocessing/data_count.ipynb): Notebook que
  muestra los datos de los que se dispone inicialmente sin preprocesar.
- [audio_preprocessing.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/preprocessing/audio_preprocessing.ipynb):
  Notebook que muestra las técnicas empleadas para el preprocesado de datos de audio.
- [image_preprocessing.py](https://github.com/dainelli98/tfg-tinyml/blob/main/preprocessing/image_preprocessing.py):
  Módulo de Python que contiene las funciones que realizan el preprocesado de los datos de imagen.
- [image_preprocessing.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/preprocessing/image_preprocessing.ipynb):
  Notebook que muestra las técnicas empleadas para el preprocesado de datos de imagen.
- [data_split.py](https://github.com/dainelli98/tfg-tinyml/blob/main/preprocessing/data_split.py): Script de Python que
  divide conjunto de datos en particiones de entrenamiento y test en carpetas separadas.
## Entrenamiento de modelos
Notebooks y scripts de Python complementarios para el entrenamiento de modelos de aprendizaje automático de sonido e
imagen.
### Entrenamiento de modelos de audio
- [audio_model_training.py](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20training/audio/audio_model_training.py):
  Módulo de Python que contiene las funciones y constantes utilizadas para entrenar modelos de TensorFlow que clasifican 
  segmentos de audio.
- [audio_model_training.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20training/audio/audio_model_training.ipynb):
  Notebook que realiza el entrenamiento de modelos de TensorFlow que clasifican segmentos de audio.
### Entrenamiento de modelos de imagen
- [image_model_training.py](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20training/image/image_model_training.py):
  Módulo de Python que contiene las funciones y constantes utilizadas para entrenar modelos de TensorFlow que clasifican 
  imágenes.
- [image_model_training.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20training/image/image_model_training.ipynb):
  Notebook que realiza el entrenamiento de modelos de TensorFlow que clasifican imágenes.
## Optimización de modelos
Notebooks y scripts de Python complementarios para la optimización de modelos de aprendizaje automático de sonido e
imagen.
- [tf_lite_conversion.py](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20optimization/tf_lite_conversion.py):
  Módulo de Python que contiene las funciones utilizadas para convertir modelos de TensorFlow en modelos TensorFlow Lite
  aplicando o no cuantización.
- [tf_lite_conversion.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20optimization/tf_lite_conversion.ipynb):
  Notebook que realiza la conversión de modelos de TensorFlow en modelos TensorFlow Lite aplicando o no cuantización.
- [tf_lite_micro_conversion.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20optimization/tf_lite_micro_conversion.ipynb):
  Notebook que convierte modelos TensorFlow Lite a modelos TensorFlow Lite para microcontroladores usando flatbuffers.
### Optimización de modelos de audio
- [audio_model_optimization.py](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20optimization/audio/audio_model_optimization.py):
  Módulo de Python que contiene las funciones y constantes empleadas para la optimización de modelos de audio.
- [audio_model_pruning.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20optimization/audio/audio_model_pruning.ipynb):
  Notebook que aplica pruning a un modelo de audio para generar un modelo optimizado.
- [audio_model_quantization_aware_training.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20optimization/audio/audio_quantization_aware_training.ipynb):
  Notebook que reentrena un modelo de audio usando quantization-aware training para generar un modelo optimizado.
### Optimización de modelos de imagen
- [image_model_optimization.py](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20optimization/image/image_model_optimization.py):
  Módulo de Python que contiene las funciones y constantes empleadas para la optimización de modelos de imagen.
- [image_model_pruning.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20optimization/image/image_model_pruning.ipynb):
  Notebook que aplica pruning a un modelo de imagen para generar un modelo optimizado.
- [image_model_quantization_aware_training.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20optimization/image/image_quantization_aware_training.ipynb):
  Notebook que reentrena un modelo de imagen usando quantization-aware training para generar un modelo optimizado.
## Evaluación de modelos
Notebooks y scripts de Python que realizan diversas pruebas y análisis sobre modelos de imagen y audio.
### Evaluación de modelos de audio
- [audio_model_evaluation.py](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20evaluation/audio/audio_model_evaluation.py):
  Módulo de Python que contiene todas las funciones que se utilizarán en la evaluación de modelos de audio.
- [audio_model_initial_evaluation.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20evaluation/audio/audio_model_initial_evaluation.ipynb):
  Notebook que permite evaluar el rendimiento de un modelo de audio sobre un conjunto de test.
- [audio_data_origin_analysis.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20evaluation/audio/audio_data_origin_analysis.ipynb):
  Notebook donde se compara el rendimiento de dos modelos de audio con datos externos y captados directamente con
  microcontrolador sobre datos de los 2 tipos.
- [audio_tf_lite_initial_evaluation.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20evaluation/audio/audio_tf_lite_initial_evaluation.ipynb):
  Notebook que permite evaluar el rendimiento de un modelo de audio convertido a TensorFlow Lite sobre un conjunto de
  test.
- [audio_optimization_analysis.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20evaluation/audio/audio_optimization_analysis.ipynb):
  Notebook que permite evaluar la variación de rendimiento que sufren modelos de audio convertidos a TensorFlow Lite
  dependiendo de la optimización aplicada.
### Evaluación de modelos de imagen
- [image_model_evaluation.py](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20evaluation/image/image_model_evaluation.py):
  Módulo de Python que contiene todas las funciones que se utilizarán en la evaluación de modelos de imagen.
- [image_model_initial_evaluation.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20evaluation/image/image_model_initial_evaluation.ipynb):
  Notebook que permite evaluar el rendimiento de un modelo de imagen sobre un conjunto de test mostrando ejemplos 
  individuales.
- [image_data_origin_analysis.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20evaluation/image/image_data_origin_analysis.ipynb):
  Notebook donde se compara el rendimiento de dos modelos de imagen con datos externos y captados directamente con 
  microcontrolador sobre datos de los 2 tipos.
- [image_tf_lite_initial_evaluation.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20evaluation/image/image_tf_lite_intial_evaluation.ipynb):
  Notebook que permite evaluar el rendimiento de un modelo de imagen convertido a TensorFlow Lite sobre un conjunto de
  test.
- [image_optimization_analysis.ipynb](https://github.com/dainelli98/tfg-tinyml/blob/main/model%20evaluation/image/image_optimization_analysis.ipynb):
  Notebook que permite evaluar la variación de rendimiento que sufren modelos de imagen convertidos a TensorFlow Lite
  dependiendo de la optimización aplicada.
## Aplicaciones TinyML
- [audio_tinyml_app](https://github.com/dainelli98/tfg-tinyml/tree/main/tinyml/audio_tinyml_app): Esta carpeta contiene
  el código fuente de una aplicación TinyML que utiliza el modelo de audio entrenado para realizar predicciones sobre el
  audio captado con el micrófono del microcontrolador.
- [image_tinyml_app](https://github.com/dainelli98/tfg-tinyml/tree/main/tinyml/image_tinyml_app): Esta carpeta contiene
  el código fuente de una aplicación TinyML que utiliza el modelo de imagen entrenado para realizar predicciones sobre 
  las capturas realizadas con el módulo de cámara conectado al microcontrolador.
- [multi_tenant_tinyml_app](https://github.com/dainelli98/tfg-tinyml/tree/main/tinyml/multi_tenant_tinyml_app): Esta
  carpeta contiene el código fuente de una aplicación TinyML que aplica un esquema de multi-tenancy para realizar una
  función de control de aforo.
## Modelos guardados
En la carpeta [saved models](https://github.com/dainelli98/tfg-tinyml/tree/main/saved%20models) se encuentran los
modelos que se han generado en le proyecto y algunos de los datos relacionados con su entrenamiento y optimización.
