import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from typing import Any

from image_model_training import COLOR_MODE, SEED, IMG_SIZE, normalize_dataset
from audio_model_evaluation import get_dataset as get_audio_dataset


def convert_saved_model(model_name: str, model_dir: str,  output_dir: str, quantize=False, representative_dataset=None):
    """
    Convierte un modelo TensorFlow en un modelo TensorFlow Lite cuantizándolo o no.
    Args:
        model_name: str con el nombre del modelo.
        model_dir:  str con el directorio donde está guardado el modelo.
        output_dir: str con el directorio donde se guardará el modelo.
        quantize:   bool que indica si se debe cuantizar el modelo.
        representative_dataset: Any dataset que sirve para ajustar la cuantización.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(f"{model_dir}/{model_name}")

    if quantize:
        print(f"Applying quantization to {model_name}.")
        output_path = f"{output_dir}/{model_name}Quant.tflite"
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        output_path = f"{output_dir}/{model_name}.tflite"

    tf_lite_model = converter.convert()

    with open(output_path, "wb") as output_file:
        output_file.write(tf_lite_model)

    print(f"Model converted. The model has been saved in {output_path}.")


def get_image_representative_dataset(data_dir: str, normalize=False) -> Any:
    """
    Genera un dataset representativo de imágenes a partir de los datos alojados en data_dir.
    Args:
        data_dir:   str path del directorio donde se alojan las imágenes que se quieren usar para crear el dataset
        normalize:  bool que indica si normalizar los datos.

    Returns:
        Any dataset que se ha generado con los datos del directorio indicado.
    """
    dataset = image_dataset_from_directory(data_dir, seed=SEED, image_size=IMG_SIZE, batch_size=100,
                                           color_mode=COLOR_MODE)

    if normalize:
        normalize_dataset(dataset)

    def representative_dataset():
        for data, _ in dataset.batch(1).take(100):
            for sample in data:
                yield [tf.dtypes.cast(sample, tf.float32)]

    return representative_dataset


def get_audio_representative_dataset(data_dir: str) -> Any:
    """
    Genera un dataset representativo de imágenes a partir de los datos alojados en data_dir.
    Args:
        data_dir:   str path del directorio donde se alojan las imágenes que se quieren usar para crear el dataset

    Returns:
        Any dataset que se ha generado con los datos del directorio indicado.
    """
    dataset = get_audio_dataset(data_dir, prefetch=False)

    def representative_dataset():
        for data, _ in dataset.batch(1).take(100):
            yield [tf.dtypes.cast(data, tf.float32)]

    return representative_dataset

