from typing import Any
import sys
import tensorflow as tf
import pathlib
import numpy as np
import random as rnd
PROJECT_DIR = project_dir = "/home/daniel/PycharmProjects/tfg-tinyml"
sys.path.insert(1, f"{project_dir}/model training/audio")

from audio_model_training import preprocess_dataset

# Seed asociada al entrenamiento.
SEED = 135209


def get_test_dataset(data_dir: str) -> Any:
    """
    Crea un dataset con los audios de test alojados en data_dir.
    Args:
        data_dir:   str con el path a la carpeta que contiene las muestras de audio de la partición test.

    Returns:
        dataset con los audios y las labels del dataset de test.
    """
    data_dir = pathlib.Path(data_dir)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rnd.seed(SEED)

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)

    nsamples = len(filenames)

    print(f"Se usarán {nsamples} muestras.")

    test_dataset = preprocess_dataset(filenames, batch=False)

    return test_dataset
