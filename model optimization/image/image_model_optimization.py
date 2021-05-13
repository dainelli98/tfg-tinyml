import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import clone_model
from tensorflow.keras import layers

from image_model_training import get_image_model, get_image_datasets

# Pruning
PRUNING_EPOCHS: int = 500
PRUNABLE_LAYERS = [layers.ZeroPadding2D, layers.Conv2D, layers.BatchNormalization, layers.ReLU, layers.Flatten,
                   layers.Dense]

PATIENCE: int = 20


def apply_prunning(model):

    def apply_pruning_to_layer(layer):
        for prunable_layer in PRUNABLE_LAYERS:
            if isinstance(layer, prunable_layer):
                return tfmot.sparsity.keras.prune_low_magnitude(layer)
        return layer

    pruned_model = clone_model(model, clone_function=apply_pruning_to_layer)

    return pruned_model
