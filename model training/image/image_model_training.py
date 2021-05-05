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

# Complejidad del modelo
COMPLEX = 0
TINYML = 1

# Tamaño de bloque de muestras.
BATCH_SIZE = 32

# Resolución de las imágenes.
IMG_SIZE = (96, 96)
IMG_SHAPE = (96, 96, 1)

# Seed asociada al entrenamiento.
SEED = 13524

# Ajuste de los datasets
AUTOTUNE = tf.data.experimental.AUTOTUNE


def augment_data(dataset: Any) -> Any:
    """
    Aplica data augmentation a un dataset y devuelve el resultado. Consiste en aplicar pequeñas rotaciones, zoom o hacer
    flips horizontales de la imagen.
    Args:
        dataset:    Any dataset al que se aplica el data augmentation.

    Returns:
        Any dataset con data augmentation aplicado.

    """
    data_augmentation = Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=IMG_SHAPE),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1)
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


def get_image_datasets(data_dir: str, validation_percentage: float, test_percentage: float,
                       shuffle=True) -> (Any, Any, Any):
    """
    Devuelve los datasets de entrenamiento, validación y test que se pueden usar para entrenar modelos aprendizaje
    automático.
    Args:
        data_dir:               str con el path del directorio donde se almacenan las imágenes separadas en carpetas con
                                el nombre de la clase a la que pertenecen.
        validation_percentage:  float con el porcentaje de muestras que se añadirán al dataset de validación.
        test_percentage:        float con el porcentaje de muestras que se añadirán al dataset de test.
        shuffle:                bool que indica si se debe aplicar shuffle a las muestras.

    Returns:
        (Any, Any, Any) con los tres datasets de imágenes de entrenamiento, validación y test.
    """
    train_dataset = image_dataset_from_directory(data_dir, shuffle=shuffle, batch_size=BATCH_SIZE, image_size=IMG_SIZE,
                                                 subset="training",
                                                 validation_split=((validation_percentage + test_percentage) / 100),
                                                 seed=SEED)
    validation_dataset = image_dataset_from_directory(data_dir, shuffle=shuffle, batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE, subset="validation",
                                                      validation_split=((validation_percentage + test_percentage)
                                                                        / 100),
                                                      seed=SEED)
    batches = tf.data.experimental.cardinality(validation_dataset)
    test_batches = batches // int((validation_percentage + test_percentage) / test_percentage)
    test_dataset = validation_dataset.take(test_batches)
    validation_dataset = validation_dataset.skip(test_batches)

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


def data_augmentation_demo(dataset: Any, data_augmentation: Any):
    """
    Muestra el resultado de aplicar data augmentation en una imagen.
    Args:
        dataset:            Any con el dataset del que se extraen las imágenes de la demo.
        data_augmentation:  Any con el data augmentation que se aplica.
    """
    for image, _ in dataset.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')


def get_model(alpha: float, depth_multiplier: float, preprocess_input: Any, train_dataset: Any,
              data_augmentation=None) -> Any:
    """
    Devuelve un modelo que usa transfer learning con MobileNet V1 con alpha y depth_multiplier indicados y las capas de
    preprocesado ofrecidas.
    Args:
        alpha:              float que controla la amplitud del modelo MobileNet usado.
        depth_multiplier:   float con el depth_multiplier que se aplica a la convolución depthwise de MobileNet.
        preprocess_input:   Any capas de TensorFlow con el preprocesado que se aplica a las imágenes.
        train_dataset:      Any con el dataset de entrenamiento.
        data_augmentation:  Any con las layers de TensorFlow que aplican data augmentation.

    Returns:
        Any modelo que usa transfer learning con MobileNet V1 con alpha y depth_multiplier indicados y las capas de
        preprocesado ofrecidas.
    """
    # Creamos el modelo MobileNet que se usará.
    mobilenet = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE, include_top=False, weights=None,
                                                alpha=alpha, depth_multiplier=depth_multiplier)
    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = mobilenet(image_batch)
    print(feature_batch.shape)
    mobilenet.trainable = True
    mobilenet.summary()

    # Creamos las capas de clasificación.
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)
    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    # Unimos las partes de la red.
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    if data_augmentation:
        augmentation_layers = data_augmentation(inputs)
        preprocess_layers = preprocess_input(augmentation_layers)
    else:
        preprocess_layers = preprocess_input(inputs)
    mobilenet_layers = mobilenet(preprocess_layers, training=True)
    average_layers = global_average_layer(mobilenet_layers)
    dropout_layers = tf.keras.layers.Dropout(0.2)(average_layers)
    output_layers = prediction_layer(dropout_layers)
    final_model = tf.keras.Model(inputs, output_layers)

    return final_model

