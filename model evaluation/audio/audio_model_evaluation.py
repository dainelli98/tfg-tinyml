from typing import Any, List
import tensorflow as tf
import pathlib
import random as rnd
from tensorflow.keras.models import load_model
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import os

from audio_model_training import preprocess_dataset, SEED

MICRO: int = 0
EXT: int = 1
PRUN: int = 2

DIGITS: int = 5


def get_dataset(data_dir: str, prefetch=True) -> Any:
    """
    Crea un dataset con los audios de alojados en data_dir.
    Args:
        data_dir:   str con el path a la carpeta que contiene las muestras de audio de la partición test.
        prefetch:   bool que indica si se aplica prefetch.

    Returns:
        dataset con los audios y las labels del dataset.
    """
    data_dir = pathlib.Path(data_dir)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rnd.seed(SEED)

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)

    nsamples = len(filenames)

    print(f"Using {nsamples} samples.")

    dataset = preprocess_dataset(filenames, batch=False, prefetch=prefetch)

    return dataset


def tensorflow_lite_model_evaluation(model_path: str, test_dirs: List[str], class_names_path: str, quantized=False):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    print(f'Testing model located in "{model_path}".')

    print(f"Model size {os.path.getsize(model_path) / 1024} Kb")

    class_names = load(class_names_path)

    for test_dir in test_dirs:
        tensorflow_lite_test_audio(interpreter, test_dir, class_names, quantized=quantized)

    if quantized:
        print("Quantized models perform slower as they are intended to work on ARM devices.")
    

def tensorflow_lite_predict(interpreter, test_dataset, quantized=False):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    true_labels = []
    times = []

    for spectrogram, label in test_dataset:
        true_labels.append(label)

        if quantized:
            input_scale, input_zero_point = input_details["quantization"]
            spectrogram = spectrogram / input_scale + input_zero_point
        spectrogram = np.expand_dims(spectrogram, axis=0).astype(input_details["dtype"])

        interpreter.set_tensor(input_details["index"], spectrogram)

        t_ini = time.time()
        interpreter.invoke()
        t_end = time.time()
        elapsed = (t_end - t_ini) * 1000  # ms
        times.append(elapsed)

        prediction = interpreter.get_tensor(output_details["index"])[0]

        predicted_class = prediction.argmax()
        predictions.append(predicted_class)

    return predictions, true_labels, times


def tensorflow_lite_test_audio(interpreter: Any, test_dir: str, class_names, quantized=False):

    test_dataset = get_dataset(test_dir, prefetch=False)
    test_dataset = test_dataset.as_numpy_iterator()

    predictions, true_labels, times = tensorflow_lite_predict(interpreter, test_dataset, quantized=quantized)

    show_test_results(true_labels, predictions, class_names)

    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    print(f"Average time: {avg_time} ms\nMax time: {max_time} ms\nMin time: {min_time} ms")


def tensorflow_model_evaluation(model_path: str, class_names_path: str, test_dirs: List[str]):
    """
    Comprueba el rendimiento de un modelo TensorFlow de audio.
    Args:
        model_path:         str con el path de la carpeta que guarda el modelo de TensorFlow que se usará para el test.
        class_names_path:   str con el path del archivo que guarda la lista de los nombres de las clases del modelo.
        test_dirs:          List[str] con los paths de los directorios con las imágenes que se usarán para sucesivos
                            tests.
    """
    model = load_model(model_path)
    class_names = load(class_names_path)
    print(f'Testing model located in "{model_path}".')
    for test_dir in test_dirs:
        tensorflow_test_audio(model, test_dir, class_names)


def get_audios_and_labels(data_dir: str) -> (List[Any], List[int]):
    """
    Obtiene los audios y labels de un directorio.
    Args:
        data_dir:   str path al directorio que contiene los audios.

    Returns:
        List[Any], List[int] con los audios y las labels.
    """
    test_dataset = get_dataset(data_dir)
    test_audios = []
    test_labels = []
    for audio, label in test_dataset:
        test_audios.append(audio.numpy())
        test_labels.append(label.numpy())
    test_audios = np.array(test_audios)
    test_labels = np.array(test_labels)
    return test_audios, test_labels


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


def tensorflow_test_audio(model: Any, test_dir: str, class_names: List[str]):
    """
    Realiza un test de predicción sobre los datos de un directorio.
    Args:
        model:          Any modelo usado para realizar las predicciones.
        test_dir:       str path al directorio que contiene las muestras de audio.
        class_names:    List[str] con los nombres de las clases a predecir.
    """
    print(f'Testing model with files located in "{test_dir}".')
    test_audios, test_labels = get_audios_and_labels(test_dir)
    predictions = np.argmax(model.predict(test_audios), axis=1)
    show_test_results(test_labels, predictions, class_names)
