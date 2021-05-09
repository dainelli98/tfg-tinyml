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

# Definimos algunas constantes

# Origen de los datos
MICRO = 0
EXT = 1

# Tamaño de bloque de muestras.
BATCH_SIZE = 32

# Seed asociada al entrenamiento.
SEED = 135209

# Tipo de datos
DATA_TYPE = "Audio"
SAMPLE_RATE = 16000     # Hz
FRAME_LENGTH = 255
FRAME_STEP = 128

# Ajuste de los datasets
AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = ["no", "silence", "unknown", "yes"]
NCLASSES = len(CLASS_NAMES)


def preprocess_dataset(files: Any, show_waveform_samples=False, show_spectrogram_example=False,
                       show_spectrogram_samples=False, batch=True) -> Any:
    """
    Args:
        files:                      Any labels y los paths de las muestras de audio.
        show_waveform_samples:      bool que indica si se quiere mostrar ejemplos de los waveforms de las muestras.
        show_spectrogram_example:   bool que indica si se quiere mostrar un ejemplo de conversion de waveform a
                                    espectrograma.
        show_spectrogram_samples:   bool que indica si se quiere mostrar ejemplos de los espectrogramas de las muestras.
        batch:                      bool que indica si se aplica batch.

    Returns:
        Any dataset con las labels y las muestras de audio preprocesadas en formato espectrograma.
    """
    dataset = tf.data.Dataset.from_tensor_slices(files)
    waveform_dataset = dataset.map(get_waveform_sample, num_parallel_calls=AUTOTUNE)

    if show_waveform_samples:
        show_waveforms(waveform_dataset)

    if show_spectrogram_example:
        show_spectrogram(waveform_dataset)

    spectrogram_dataset = waveform_dataset.map(get_spectrogram_sample, num_parallel_calls=AUTOTUNE)

    if show_spectrogram_samples:
        show_spectrograms(spectrogram_dataset)

    if batch:
        spectrogram_dataset = spectrogram_dataset.batch(BATCH_SIZE)

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
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    x = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    y = range(height)
    ax.pcolormesh(x, y, log_spec)


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
    plot_spectrogram(spectrogram.numpy(), axes[1])
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
        plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
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
    spectrogram = tf.signal.stft(equal_length, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)
    spectrogram = tf.abs(spectrogram)
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


def get_audio_model(input_shape: (int, int, int), model_name: str, train_dataset: Any):
    """
    Genera un modelo tiny_conv de audio con el input shape y el número de clases indicados.
    Args:
        input_shape:    (int, int, int) el input shape de los datos.
        model_name:     str con el nombre que se asigna al modelo.
        train_dataset:  Any con el dataset que contiene las muestras de entrenamiento.

    Returns:
        Any modelo resultante.
    """
    # Ajustamos la normalización en base a estadísticas del dataset de entrenamiento.
    normalization_layer = layers.experimental.preprocessing.Normalization()
    normalization_layer.adapt(train_dataset.map(lambda x, _: x))

    return Sequential([
        layers.Input(shape=input_shape),
        layers.experimental.preprocessing.Resizing(32, 32),
        normalization_layer,
        layers.Conv2D(8, (8, 10), strides=(2, 2), activation=tf.nn.relu6),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(NCLASSES)
    ], name=model_name)
