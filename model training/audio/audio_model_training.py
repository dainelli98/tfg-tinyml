import tensorflow as tf
import numpy as np
from typing import Any
import random as rnd
import os
import matplotlib.pyplot as plt
from IPython import display
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import pathlib
import tensorflow_io as tfio
from librosa.feature import melspectrogram
from librosa.core import power_to_db
from matplotlib import cm

# Definimos algunas constantes

# Origen de los datos
MICRO: int = 0
EXT: int = 1

# Tamaño de bloque de muestras.
BATCH_SIZE: int = 32

# Seed asociada al entrenamiento.
SEED: int = 135209

# Tipo de datos
DATA_TYPE: str = "Audio"
SAMPLE_RATE: int = 16000     # Hz
N_FFT: int = 512
HOP_LENGTH: int = 320
WINDOW_LENGTH: int = 480
N_MELS: int = 40

# Ajuste de los datasets
AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES: [str] = ["no", "silence", "unknown", "yes"]
NCLASSES: int = len(CLASS_NAMES)


def preprocess_dataset(files: Any, show_waveform_samples=False, show_spectrogram_example=False,
                       show_spectrogram_samples=False, batch=True, prefetch=True) -> Any:
    """
    Args:
        files:                      Any labels y los paths de las muestras de audio.
        show_waveform_samples:      bool que indica si se quiere mostrar ejemplos de los waveforms de las muestras.
        show_spectrogram_example:   bool que indica si se quiere mostrar un ejemplo de conversion de waveform a
                                    espectrograma.
        show_spectrogram_samples:   bool que indica si se quiere mostrar ejemplos de los espectrogramas de las muestras.
        batch:                      bool que indica si se aplica batch.
        prefetch:                   bool que indica si se aplica prefetch.

    Returns:
        Any dataset con las labels y las muestras de audio preprocesadas en formato espectrograma.
    """
    dataset = tf.data.Dataset.from_tensor_slices(files)
    waveform_dataset = dataset.map(get_waveform_sample, num_parallel_calls=AUTOTUNE)

    if show_waveform_samples:
        show_waveforms(waveform_dataset)

    if show_spectrogram_example:
        show_spectrogram(waveform_dataset)

    spectrogram_dataset = waveform_dataset.map(lambda waveform, class_name: tf.py_function(func=get_spectrogram_sample,
                                                                                           inp=[waveform, class_name],

                                                                                           Tout=[tf.float32, tf.int64]))

    def set_spectrogram_shape(data: Any, label: int) -> (Any, int):
        """
        Ajusta las shapes de los datasets después de usar py_function.
        Args:
            data:   Any con los datos de espectrograma.
            label:  int con la label de los datos.

        Returns:
            (Any, int) con los datos con la shape correcta.
        """
        data.set_shape([40, 49, 1])
        label.set_shape([])
        return data, label

    spectrogram_dataset = spectrogram_dataset.map(set_spectrogram_shape, num_parallel_calls=AUTOTUNE)

    if show_spectrogram_samples:
        show_spectrograms(spectrogram_dataset)

    if batch:
        spectrogram_dataset = spectrogram_dataset.batch(BATCH_SIZE)

    if prefetch:
        spectrogram_dataset = spectrogram_dataset.cache().prefetch(AUTOTUNE)

    return spectrogram_dataset


def get_audio_datasets(data_dir: str, validation_percentage: float, show_waveform_samples=False,
                       show_spectrogram_example=False, show_spectrogram_samples=False) -> (Any, Any):
    """
    Devuelve los datasets de entrenamiento, validación y test que se pueden usar para entrenar modelos aprendizaje
    automático.
    Args:
        data_dir:                   str con el path del directorio donde se almacenan los archivos de sonido separadas
                                    en carpetas con el nombre de la clase a la que pertenecen.
        validation_percentage:      float con el porcentaje de muestras que se añadirán al dataset de validación.
        show_waveform_samples:      bool que indica si se quiere mostrar ejemplos de los waveforms de las muestras.
        show_spectrogram_example:   bool que indica si se quiere mostrar un ejemplo de conversion de waveform a
                                    espectrograma.
        show_spectrogram_samples:   bool que indica si se quiere mostrar ejemplos de los espectrogramas de las muestras.

    Returns:
        (Any, Any) con los datasets train y validation.
    """
    data_dir = pathlib.Path(data_dir)

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rnd.seed(SEED)

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)

    nsamples = len(filenames)
    nvalidation_samples = int(nsamples * (validation_percentage / 100))
    ntrain_samples = nsamples - nvalidation_samples

    print(f"Se usarán {nsamples} muestras:\n\t- Train: {ntrain_samples}\n\t- Validation: {nvalidation_samples}")

    train_files = filenames[:ntrain_samples]
    validation_files = filenames[ntrain_samples:]

    train_dataset = preprocess_dataset(train_files, show_waveform_samples=show_waveform_samples,
                                       show_spectrogram_example=show_spectrogram_example,
                                       show_spectrogram_samples=show_spectrogram_samples)
    validation_dataset = preprocess_dataset(validation_files)

    return train_dataset, validation_dataset


def plot_spectrogram(spectrogram: Any, ax: Any):
    """
    Muestra un espectrograma en ax.
    Args:
        spectrogram:    Any espectrograma que se quiere mostrar.
        ax:             Any ax donde se quiere mostrar el espectrograma.
    """
    ax.imshow(np.swapaxes(spectrogram, 0, 1), interpolation='nearest', cmap=cm.viridis, origin='lower',
              aspect='auto')


def show_spectrogram(dataset: Any):
    """
    Dado un dataset de muestras de audio muestra un ejemplo de conversion de waveform a espectrograma.
    Args:
        dataset:    Any dataset del que se muestra el ejemplo.
    """
    label = spectrogram = waveform = None
    for waveform, label in dataset.take(1):
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(waveform)

    print(f"Ejemplo de conversion de waveform a espectrograma:\n\t- Label: {label}\n\t- Waveform shape: "
          f"{waveform.shape}\n\t- Spectrogram shape: {spectrogram.shape}\n\t")
    print("Audio playback")
    # noinspection PyTypeChecker
    display.display(display.Audio(waveform, rate=16000))
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])
    plot_spectrogram(spectrogram, axes[1])
    axes[1].set_title('Spectrogram')
    plt.show()


def show_waveforms(dataset: Any, rows=3, cols=3):
    """
    Dado un dataset de muestras de audio muestra los waveforms de algunas muestras.
    Args:
        dataset:    Any dataset del que se muestran las muestras de ejemplo.
        rows:       int con el número de filas de muestras que se quieren mostrar.
        cols:       int con el número de columnas de muestras que se quieren mostrar.
    """
    print(f"Ejemplos de muestras en formato waveform.")
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
    for i, (audio, label) in enumerate(dataset.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(audio.numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        label = label.numpy().decode('utf-8')
        ax.set_title(label)
    plt.show()


def show_spectrograms(dataset: Any, rows=3, cols=3):
    """
    Dado un dataset de muestras de audio muestra los waveforms de algunas muestras.
    Args:
        dataset:        Any dataset del que se muestran las muestras de ejemplo.
        rows:           int con el número de filas de muestras que se quieren mostrar.
        cols:           int con el número de columnas de muestras que se quieren mostrar.
    """
    print(f"Ejemplos de muestras en formato espectrograma.")
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, (spectrogram, label_id) in enumerate(dataset.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(np.squeeze(spectrogram), ax)
        ax.set_title(CLASS_NAMES[label_id.numpy()])
        ax.axis('off')
    plt.show()


def decode_audio_file(audio_file: Any) -> Any:
    """
    Obtiene los datos de audio de un archivo wav.
    Args:
        audio_file: Any con el archivo del que tiene datos de audio.

    Returns:
        Any con los datos del archivo de audio indicado.
    """
    waveform, sr_in = tf.audio.decode_wav(audio_file)
    waveform = tfio.audio.resample(waveform, tf.cast(sr_in, tf.int64), SAMPLE_RATE)
    return tf.squeeze(waveform, axis=-1)


def get_label(filepath: Any) -> str:
    """
    Obtiene la label de una muestra a partir de su file tensor.
    Args:
        filepath:   Any file tensor de la muestra de la que se quiere obtener la label.

    Returns:
        str con la label de la muestra.
    """
    parts = tf.strings.split(filepath, os.path.sep)
    return parts[-2]


def get_waveform_sample(filepath: Any) -> (Any, str):
    """
    Obtiene los datos y la label de una muestra a partir de su file tensor.
    Args:
        filepath:       Any file tensor de la muestra de la que se quiere obtener la label.

    Returns:
        (Any, str) con los datos y la label de una muestra.
    """
    label = get_label(filepath)
    audio_file = tf.io.read_file(filepath)
    waveform = decode_audio_file(audio_file)
    return waveform, label


def get_spectrogram(waveform: Any) -> Any:
    """
    A partir de unos datos de audio en formato waveform obtiene su espectrograma.
    Args:
        waveform:       Any con los datos en formato waveform de una muestra de audio.

    Returns:
        Any espectrograma de la muestra de audio.
    """
    # noinspection PyUnresolvedReferences
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)

    spectrogram = melspectrogram(equal_length.numpy(), sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                 win_length=WINDOW_LENGTH, center=False, n_mels=40)
    spectrogram = power_to_db(spectrogram, ref=np.max)
    return spectrogram


def get_spectrogram_sample(waveform: Any, label: str) -> (Any, int):
    """
    A partir de un waveform y su label obtiene label_id y
    Args:
        waveform:       Any con el waveform de la muestra.
        label:          str con el nombre de la clase asociada a la muestra.

    Returns:
        (Any, int) con el espectrograma y la label_id de una muestra.

    """
    spectrogram = get_spectrogram(waveform)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == CLASS_NAMES)
    return spectrogram, label_id


def prepara_data_for_normalization_adapt(dataset: Any) -> Any:
    """
    Prepara los datos de un dataset para ser usados para ajustar la normalización de una capa.
    Args:
        dataset:    Any dataset del que se extraen los datos.

    Returns:
        Any datos del dataset listos para ser usados para ajustar una capa de normalización.
    """
    data_list = [x for x, y in list(dataset.as_numpy_iterator())]
    data = data_list[0]
    try:
        for batch in data_list[1:]:
            data = np.concatenate((data, batch), axis=0)
    except Exception:
        print("Dataset only had 1 batch")
    return data


def get_audio_model(input_shape: (int, int, int), model_name: str, train_dataset: Any, normalize=True):
    """
    Genera un modelo tiny_conv de audio con el input shape y el número de clases indicados.
    Args:
        input_shape:    (int, int, int) el input shape de los datos.
        model_name:     str con el nombre que se asigna al modelo.
        train_dataset:  Any con el dataset que contiene las muestras de entrenamiento.
        normalize:      bool que indica si se añade una capa de normalización.

    Returns:
        Any modelo resultante.
    """
    if normalize:
        # Ajustamos la normalización en base a estadísticas del dataset de entrenamiento.
        normalization_layer = layers.experimental.preprocessing.Normalization()

        # Hay que obtener los datos en un formato que sirva de input para adapt
        data = prepara_data_for_normalization_adapt(train_dataset)

        normalization_layer.adapt(data)

        capas = [
            layers.InputLayer(input_shape=input_shape),
            normalization_layer,
            layers.Conv2D(8, (8, 10), strides=(2, 2), activation=tf.nn.relu6),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(NCLASSES)
        ]

    else:
        capas = [
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(8, (8, 10), strides=(2, 2), activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(NCLASSES)
        ]

    return Sequential(capas, name=model_name)
