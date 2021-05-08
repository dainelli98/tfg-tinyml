from typing import Any
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Definimos algunas constantes

# Origen de los datos
MICRO = 0
EXT = 1

# Tamaño de bloque de muestras.
BATCH_SIZE = 32

# Resolución de las imágenes.
IMG_SIZE = (96, 96)
IMG_SHAPE = (96, 96, 1)

# Seed asociada al entrenamiento.
SEED = 13524

# Ajuste de los datasets
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Tipo de datos
DATA_TYPE = "Img"
COLOR_MODE = "grayscale"


def augment_data(dataset: Any, image_shape: (int, int, int)) -> Any:
    """
    Aplica data augmentation a un dataset y devuelve el resultado. Consiste en aplicar pequeñas rotaciones, zoom o hacer
    flips horizontales de la imagen.
    Args:
        dataset:        Any dataset al que se aplica el data augmentation.
        image_shape:    (int, int, int) con las dimensiones de las imágenes que se quieren aumentar.

    Returns:
        Any dataset con data augmentation aplicado.
    """
    data_augmentation = Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=image_shape),
        layers.experimental.preprocessing.RandomRotation(0.05),
        layers.experimental.preprocessing.RandomZoom(0.05)
    ])
    augmented_dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    augmented_dataset = augmented_dataset.shuffle(1000, seed=SEED)
    return augmented_dataset


def normalize_dataset(dataset: Any, negative=True) -> Any:
    """
    Normaliza un dataset con valores en [0, 255] para que el rango de valores resultante sea [-1, 1] o [0, 1].
    Args:
        dataset:    Any dataset que se quiere normalizar.
        negative:   bool que indica si el rango final debe ser [-1, 1] o [0, 1]

    Returns:
        Any dataset normalizado.
    """
    if negative:
        normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)
    else:
        normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255.)
    normalized_dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    return normalized_dataset


def get_image_datasets(data_dir: str, validation_percentage: float, shuffle=True) -> (Any, Any, Any):
    """
    Devuelve los datasets de entrenamiento, validación y test que se pueden usar para entrenar modelos aprendizaje
    automático.
    Args:
        data_dir:               str con el path del directorio donde se almacenan las imágenes separadas en carpetas con
                                el nombre de la clase a la que pertenecen.
        validation_percentage:  float con el porcentaje de muestras que se añadirán al dataset de validación.
        shuffle:                bool que indica si se debe aplicar shuffle a las muestras del dataset de entrenamiento.

    Returns:
        (Any, Any, Any) con los 2 datasets de imágenes de entrenamiento y validación.
    """
    train_dataset = image_dataset_from_directory(data_dir, validation_split=(validation_percentage / 100.),
                                                 shuffle=True, subset="training", seed=SEED, image_size=IMG_SIZE,
                                                 batch_size=BATCH_SIZE, color_mode=COLOR_MODE)

    validation_dataset = image_dataset_from_directory(data_dir, validation_split=(validation_percentage / 100.),
                                                      shuffle=True, subset="validation", seed=SEED, image_size=IMG_SIZE,
                                                      batch_size=BATCH_SIZE, color_mode=COLOR_MODE)

    return train_dataset, validation_dataset


def demo_image_dataset(dataset: Any, class_names=None):
    """
    Muestra un subconjunto de un dataset que sirve como ejemplo.
    Args:
        dataset:            Any con el dataset del que se extraen las imágenes de la demo.
        class_names:        [str] con los nombres de las clases ordenados por el índice que los representa en el
                            dataset.
    """
    fig, _ = plt.subplots(nrows=3, ncols=3)
    fig.set_size_inches(10, 10)
    fig.suptitle("Dataset samples")
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = fig.axes[i]
            ax.imshow(images[i].numpy().astype("uint8"), cmap='gray', vmin=0, vmax=255)
            if class_names:
                ax.set_title(class_names[labels[i]])
            ax.axis("off")


def get_image_model(input_shape: (int, int, int), nclasses: int, model_name: str):
    """
    Genera un modelo simple de imagen con el input shape y el número de clases indicados.
    Args:
        input_shape:    (int, int, int) con las dimensiones de las imágenes que ha de clasificar el modelo.
        nclasses:       int con el número de classes distintas presentes en las imágenes.
        model_name:     str con el nombre que se asigna al modelo.

    Returns:
        Any modelo resultante.
    """
    return Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=input_shape),    # Normalization
        layers.Conv2D(4, kernel_size=3, strides=2, activation=tf.nn.relu6),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(8, kernel_size=3, strides=1, activation=tf.nn.relu6),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.ZeroPadding2D(),
        layers.Conv2D(16, kernel_size=3, strides=2, activation=tf.nn.relu6),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(16, kernel_size=1, strides=1, padding="same", activation=tf.nn.relu6),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.ZeroPadding2D(),
        layers.Conv2D(32, kernel_size=3, strides=2, activation=tf.nn.relu6),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, kernel_size=1, strides=1, padding="same", activation=tf.nn.relu6),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.ZeroPadding2D(),
        layers.Conv2D(64, kernel_size=2, strides=2, padding="same", activation=tf.nn.relu6),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, kernel_size=1, strides=1, padding="same", activation=tf.nn.relu6),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(nclasses)
    ], name=model_name)
