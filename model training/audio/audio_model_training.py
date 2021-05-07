import tensorflow as tf
import numpy as np
from typing import List, Tuple, Any
import random as rnd
import os
import tensorflow_io as tfio

# Definimos algunas constantes

# Origen de los datos
MICRO = 0
EXT = 1

# Tamaño de bloque de muestras.
BATCH_SIZE = 32

# Seed asociada al entrenamiento.
SEED = 13524

# Tipo de datos
DATA_TYPE = "Sound"
SAMPLE_RATE = 16000     # Hz

tf.random.set_seed(SEED)
np.random.seed(SEED)
rnd.seed(SEED)


def get_audio_files(data_dir: str, validation_percentage: float, class_names: List[str],
                    shuffle=True) -> (List[Tuple[int, str]], List[Tuple[int, str]]):
    """
    Devuelve los datasets de entrenamiento, validación y test que se pueden usar para entrenar modelos aprendizaje
    automático.
    Args:
        data_dir:               str con el path del directorio donde se almacenan los archivos de sonido separadas en
                                carpetas con el nombre de la clase a la que pertenecen.
        validation_percentage:  float con el porcentaje de muestras que se añadirán al dataset de validación.
        class_names:            List[str] con los nombres de las clases.
        shuffle:                bool que indica si se debe aplicar shuffle a las muestras de entrenamiento.

    Returns:
        (List[Tuple[int, str]], List[Tuple[int, str]]) con las listas de muestras de entrenamiento y validación.
    """

    class_indexes = {}
    for index, name in enumerate(class_names):
        class_indexes[name] = index

    train_files = []
    validation_files = []

    print(f"\nParticionando datos:\n\t- Directorio: {data_dir}\n\t- Porcetage validación: {validation_percentage}%")
    for name in class_names:
        files = os.listdir(f"{data_dir}/{name}")
        train_count = 0
        validation_count = 0
        rnd.shuffle(files)
        nfiles = len(files)
        test_samples = nfiles * (validation_percentage / 100.)
        train_samples = nfiles - test_samples

        for file in files:

            prob = rnd.random()

            if validation_count >= test_samples or (
                    train_count < train_samples and prob > (validation_percentage / 100.)):
                train_files.append((class_indexes[name], f"{data_dir}/{name}/{file}"))
                train_count += 1
            else:
                validation_files.append((class_indexes[name], f"{data_dir}/{name}/{file}"))
                validation_count += 1

        print(
            f"\t- {nfiles} muestras de la clase {name} donde {train_count} son para entrenamiento y {validation_count}"
            f" son para validación.")

    if shuffle:
        np.random.shuffle(train_files)

    return train_files, validation_files


def read_audio_file(filepath, sample_rate=SAMPLE_RATE) -> Any:
    """
    Obtiene los datos de audio de un archivo wav.
    Args:
        filepath:       str con el path del archivo del que se quieren obtener los datos.
        sample_rate:    int con el sample rate que se desea que tenga la muestra.

    Returns:
        Any con los datos del archivo de audio indicado.
    """
    data, original_sample_rate = tf.audio.decode_wav(tf.io.read_file(filepath))
    if sample_rate and original_sample_rate != sample_rate:
        data = tfio.audio.resample(data, original_sample_rate, sample_rate)
    return tf.squeeze(data, axis=-1)
