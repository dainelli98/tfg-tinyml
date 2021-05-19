from typing import Any, List, Dict
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time

from image_model_training import COLOR_MODE


MICRO: int = 0
EXT: int = 1
PRUN: int = 2

DIGITS: int = 5


def create_class_indexes(class_names: List[str]) -> Dict[str, int]:
    """
    Crea un diccionario que relaciona nombres de clases con su label.
    Args:
        class_names:    List[str] con los nombres de las clases.

    Returns:
        Dict[str, int] con los indices de las classes indexados por nombre.
    """
    class_indexes = {}
    for index, name in enumerate(class_names):
        class_indexes[name] = index
    return class_indexes


def predict_img(img: Any, model: Any) -> (int, float):
    """
    Dados una imagen y un modelo predice la clase de la imagen usando dicho modelo.
    Args:
        img:    Any imagen que se quiere predecir.
        model:  Any modelo que se usa para la predicción.

    Returns:
        (int, float) con la clase predicha y la confianza de la predicción.
    """
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])
    predicted_class = np.argmax(score)
    prediction_score = np.max(score)
    return predicted_class, prediction_score


def present_img(img: Any, predicted_class: int, true_label: int, prediction_score: float, class_names: List[str]):
    """
    Muestra gráficamente el resultado de una predicción.
    Args:
        img:                Any con la imagen sobre la que se realiza la predicción.
        predicted_class:    int clase predicha para la imagen.
        true_label:         int clase de la imagen.
        prediction_score:   float confianza con la que se ha realizado la predicción.
        class_names:        List[str] con los nombres de las clases del modelo.
    """
    if predicted_class == true_label:
        text_color = "green"
    else:
        text_color = "red"
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.text(2, 15,
             f"Image class: {class_names[true_label]}\nPredicted class: {class_names[predicted_class]}"
             f"\nPrediction confidence: {(prediction_score * 100.).round(2)}%", color="black",
             bbox=dict(facecolor=text_color, alpha=0.8))
    plt.show()


def test_file(filepath: str, true_label: int, class_names: List[str], model: Any, show_counter: int,
              show_interval: int) -> int:
    """
    Realiza una predicción de un archivo de imagen.
    Args:
        filepath:       str path del archivo que se quiere predecir.
        true_label:     int label asociada a la imagen del archivo que se quiere predecir.
        class_names:    List[str] con los nombres de las clases del modelo.
        model:          Any el modelo de TensorFlow que se usará para el test.
        show_counter:   int con el número de imágenes de la misma clase predichas.
        show_interval:  int con la frecuencia con la que se muestran individualmente las imágenes testadas. Si es 0 no
                        se muestra ninguna.

    Returns:
        int con la clase predicha para la imagen del archivo indicado.
    """

    img = load_img(filepath, color_mode=COLOR_MODE)

    predicted_class, prediction_score = predict_img(img, model)

    if show_interval != 0 and (show_counter % show_interval == 0 or predicted_class != true_label):
        present_img(img, predicted_class, true_label, prediction_score, class_names)

        show_counter += 1

    return predicted_class


def test_class(class_dir: str, class_name: str, class_index: int, class_names: List[str], model: Any, true_labels=None,
               predictions=None, show_interval=0) -> (List[int], List[int]):
    """
    Realiza predicciones sobre un conjunto de imágenes pertenecientes a una misma clase.
    Args:
        class_dir:      str con el path del directorio que contiene las imágenes de la clase.
        class_name:     str con el nombre de la clase.
        class_index:    int índice asociado a la clase.
        class_names:    List[str] lista con las clases que puede predecir el modelo.
        model:          Any modelo que se usa para realizar las predicciones.
        true_labels:    List[int] con las labels de predicciones anteriores.
        predictions:    List[int] con las predicciones anteriores.
        show_interval:  int con la frecuencia con la que se muestran individualmente las imágenes testadas. Si es 0 no
                        se muestra ninguna.

    Returns:
        (List[int], List[int]) con las predicciones y las labels correspondientes.
    """
    if not true_labels:
        true_labels = []
    if not predictions:
        predictions = []
    show_counter = 0

    files = os.listdir(class_dir)
    print(f"Testing {len(files)} images from class {class_name}.")

    for file in files:
        filepath = f"{class_dir}/{file}"
        true_labels.append(class_index)
        predicted_class = test_file(filepath, class_index, class_names, model, show_counter, show_interval)
        predictions.append(predicted_class)
        show_counter += 1

    return true_labels, predictions


def test_images(model: Any, test_dir: str, class_names: List[str], show_interval=0):
    """
    Hace un test sobre un conjunto de imágenes usando el modelo indicado y muestra un resumen de los resultados.
    Args:
        model:          Any el modelo de TensorFlow que se usará para el test.
        test_dir:       str con el path donde se encuentran las imágenes que se usarán para el test.
        class_names:    List[str] con los nombres de las clases del modelo.
        show_interval:  int con la frecuencia con la que se muestran individualmente las imágenes testadas. Si es 0 no
                        se muestra ninguna.
    """
    class_indexes = create_class_indexes(class_names)

    print(f'Testing model with files located in "{test_dir}".')

    predictions = []
    true_labels = []

    for name in class_names:
        class_dir = f"{test_dir}/{name}"
        true_labels, predictions = test_class(class_dir, name, class_indexes[name], class_names, model, true_labels,
                                              predictions, show_interval)

    show_test_results(true_labels, predictions, class_names)


def show_test_results(true_labels: List[int], predictions: List[int], class_names: List[str]):
    """
    Muestra un resumen de los resultados de un test sobre un modelo.
    Args:
        true_labels:    List[int] con las labels de las muestras.
        predictions:    List[int] con las predicciones realizadas.
        class_names:    List[str] con los nombres de las clases que se han predecido.
    """
    confusion_mtx = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=class_names, yticklabels=class_names,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.title("Confusion matrix")
    plt.show()
    print(classification_report(true_labels, predictions, target_names=class_names, digits=DIGITS))


def tensorflow_model_evaluation(model_path: str, class_names_path: str, test_dirs: List[str], show_interval=0):
    """
    Comprueba el rendimiento de un modelo TensorFlow de imagen.
    Args:
        model_path:         str con el path de la carpeta que guarda el modelo de TensorFlow que se usará para el test.
        class_names_path:   str con el path del archivo que guarda la lista de los nombres de las clases del modelo.
        test_dirs:          List[str] con los paths de los directorios con las imágenes que se usarán para sucesivos
                            tests.
        show_interval:      int con la frecuencia con la que se muestran individualmente las imágenes testadas. Si es 0
                            no se muestra ninguna.
    """
    model = load_model(model_path)
    class_names = load(class_names_path)
    print(f'Testing model located in "{model_path}".')
    for test_dir in test_dirs:
        test_images(model, test_dir, class_names, show_interval=show_interval)


def tensorflow_lite_model_evaluation(model_path: str, class_names_path: str, test_dirs: List[str], quantized=False):
    """
    Comprueba el rendimiento de un modelo TensorFlow Lite.
    Args:
        model_path:         str con el path de la carpeta que guarda el modelo de TensorFlow que se usará para el test.
        test_dirs:          List[str] con los paths de los directorios con las imágenes que se usarán para sucesivos
        class_names_path:   str con el path del archivo que guarda la lista de los nombres de las clases del modelo.
                            tests.
        quantized:          bool que indica si el modelo está cuantizado.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    print(f'Testing model located in "{model_path}".')

    print(f"Model size {os.path.getsize(model_path) / 1024} Kb")

    class_names = load(class_names_path)

    for test_dir in test_dirs:
        tensorflow_lite_test_image(interpreter, class_names, test_dir)

    if quantized:
        print("Quantized models can perform slower as they are intended to work on ARM devices.")


def tensorflow_lite_test_image(interpreter: Any, class_names: List[str], test_dir: str):
    """
    Hace un test sobre un conjunto de imágenes usando el modelo indicado y muestra un resumen de los resultados.
    Args:
        interpreter:    Any el interpreter de TensorFlow que se usará para el test.
        class_names:    List[str] con los nombres de las clases del modelo.
        test_dir:       str con el path donde se encuentran las imágenes que se usarán para el test.
        quantized:      bool que indica si el modelo está cuantizado.
    """
    class_indexes = create_class_indexes(class_names)

    print(f'Testing model with files located in "{test_dir}".')

    predictions = []
    true_labels = []
    times = []

    for name in class_names:
        class_dir = f"{test_dir}/{name}"
        true_labels, predictions, times = tensorflow_lite_test_class(class_dir, name, class_indexes[name], interpreter,
                                                                     true_labels, predictions, times)

    show_test_results(true_labels, predictions, class_names)
    time_summary(times)


def time_summary(times: List[float]):
    """
    Hace un resumen a partir de los tiempos de inferencia de un test.
    Args:
        times:  List[float] con los tiempos de inferencia.
    """
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    print(f"Average time: {avg_time} ms\nMax time: {max_time} ms\nMin time: {min_time} ms")


def tensorflow_lite_test_class(class_dir: str, class_name: str, class_index: int, interpreter: Any, true_labels=None,
                               predictions=None, times=None):
    """
    Realiza predicciones sobre un conjunto de imágenes pertenecientes a una misma clase.
    Args:
        class_dir:      str con el path del directorio que contiene las imágenes de la clase.
        class_name:     str con el nombre de la clase.
        class_index:    int índice asociado a la clase.
        interpreter:    Any interpreter que se usa para realizar las predicciones.
        true_labels:    List[int] con las labels de predicciones anteriores.
        predictions:    List[int] con las predicciones anteriores.
        times:          List[float] con los tiempos anteriores.

    Returns:
        (List[int], List[int]) con las predicciones y las labels correspondientes.
    """
    if not true_labels:
        true_labels = []
    if not predictions:
        predictions = []

    files = os.listdir(class_dir)
    print(f"Testing {len(files)} images from class {class_name}.")

    for file in files:
        filepath = f"{class_dir}/{file}"
        true_labels.append(class_index)
        predicted_class, elapsed = tensorflow_lite_test_file(filepath, interpreter)
        predictions.append(predicted_class)
        times.append(elapsed)

    return true_labels, predictions, times


def tensorflow_lite_test_file(filepath: str, interpreter: Any) -> (int, float):
    """
    Realiza una predicción sobre un archivo.
    Args:
        filepath:       str con el path del archivo que contiene la imagen que se predice.
        interpreter:    Any interpreter que se usa para realizar las predicciones.

    Returns:
        (int, float) con la predicción y el tiempo empleado.
    """
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    img = load_img(filepath, color_mode=COLOR_MODE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], img_array)
    t_ini = time.time()
    interpreter.invoke()
    t_end = time.time()
    elapsed = (t_end - t_ini) * 1000  # ms

    prediction = interpreter.get_tensor(output_details["index"])[0]
    predicted_class = prediction.argmax()

    return predicted_class, elapsed
