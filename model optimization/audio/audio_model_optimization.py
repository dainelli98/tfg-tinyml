import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import clone_model
from tensorflow.keras import layers
from typing import Any

from audio_model_training import prepara_data_for_normalization_adapt

# Pruning
PRUNING_EPOCHS: int = 50
PRUNABLE_LAYERS = [layers.Conv2D, layers.Flatten, layers.Dense, layers.MaxPooling2D]
PRUNING_PATIENCE: int = 10

# Quantization
QUANT_EPOCHS: int = 250
QUANT_PATIENCE: int = 250


def apply_pruning(model: Any) -> Any:
    """
    Aplica pruning a un modelo de tensorflow.
    Args:
        model:  Model de TensorFlow al que se aplica pruning.

    Returns:
        Model de TensorFlow con pruning aplicado.
    """

    def apply_pruning_to_layer(layer: Any) -> Any:
        """
        Aplica low magnitude pruning a las capas compatibles.
        Args:
            layer:  Layer de TensorFlow a la que se le aplica pruning.

        Returns:
            Layer de TensorFlow a la que se ha aplicado pruning si es compatible.
        """
        for prunable_layer in PRUNABLE_LAYERS:
            if isinstance(layer, prunable_layer):
                return tfmot.sparsity.keras.prune_low_magnitude(layer)
        return layer

    pruned_model = clone_model(model, clone_function=apply_pruning_to_layer)

    return pruned_model


def normalize_dataset(dataset: Any) -> Any:
    """
    Normaliza los valores de un dataset.
    Args:
        dataset:    Any dataset que se quiere normalizar.

    Returns:
        Any dataset normalizado.
    """
    normalization_layer = layers.experimental.preprocessing.Normalization()
    # Hay que obtener los datos en un formato que sirva de input para adapt
    data = prepara_data_for_normalization_adapt(dataset)
    normalization_layer.adapt(data)

    normalized_dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    return normalized_dataset
