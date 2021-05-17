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

from audio_model_training import preprocess_dataset, SEED


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


def tensorflow_model_evaluation(model_path: str, class_names_path: str, test_dirs: [str]):
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
        test_audio(model, test_dir, class_names)


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


def test_audio(model: Any, test_dir: str, class_names: List[str]):
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
