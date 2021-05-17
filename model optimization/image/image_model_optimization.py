import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import clone_model
from tensorflow.keras import layers
from typing import Any

# Pruning
PRUNING_EPOCHS: int = 100
PRUNABLE_LAYERS = [layers.ZeroPadding2D, layers.Conv2D, layers.BatchNormalization, layers.ReLU, layers.Flatten,
                   layers.Dense, layers.MaxPooling2D]
PRUN_PATIENCE: int = 10

# Qunatization aware training
QUANT_EPOCHS: int = 500
QUANT_PATIENCE: int = 50


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
