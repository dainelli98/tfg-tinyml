from typing import Any, List
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def test_images(model: Any, test_dir: str, class_names: List[str], color_mode="grayscale", show_interval=0):
    """
    Hace un test sobre un conjunto de imágenes usando el modelo indicado y muestra un resumen de los resultados.
    Args:
        model:          Any el modelo de TensorFlow que se usará para el test.
        test_dir:       str con el path donde se encuentran las imágenes que se usarán para el test.
        class_names:    List[str] con los nombres de las clases del modelo.
        color_mode:     str con el tipo de color de las imágenes que se evalúan.
        show_interval:  int con la frecuencia con la que se muestran individualmente las imágenes testadas. Si es 0 no
                        se muestra ninguna.
    """
    class_indexes = {}
    for index, name in enumerate(class_names):
        class_indexes[name] = index
    print(f'Testing model with files located in "{test_dir}".')

    predictions = []
    true_labels = []

    classes = os.listdir(test_dir)
    show_counter = 0

    for name in classes:
        class_dir = f"{test_dir}/{name}"
        files = os.listdir(class_dir)
        print(f"Testing {len(files)} images from class {name}.")
        class_index = class_indexes[name]
        for file in files:
            true_labels.append(class_index)

            file_path = f"{test_dir}/{name}/{file}"

            img = load_img(file_path, color_mode=color_mode)
            img_array = img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            prediction = model.predict(img_array)
            score = tf.nn.softmax(prediction[0])

            predicted_class = np.argmax(score)
            predictions.append(predicted_class)
            prediction_score = np.max(score)

            if show_interval != 0 and (show_counter % show_interval == 0 or predicted_class != class_index):
                if predicted_class == class_index:
                    text_color = "green"
                else:
                    text_color = "red"
                plt.figure()
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
                plt.text(2, 15,
                         f"Image class: {name}\nPredicted class: {class_names[predicted_class]}\nPrediction confidence "
                         f"{(prediction_score * 100.).round(2)}%", color="black",
                         bbox=dict(facecolor=text_color, alpha=0.8))
                plt.show()

                show_counter += 1

    confusion_mtx = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=class_names, yticklabels=class_names,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.title("Confusion matrix")
    plt.show()
    print(classification_report(true_labels, predictions, target_names=class_names))


def tensorflow_model_evaluation(model_path: str, class_names_path: str, test_dirs: [str], color_mode="grayscale",
                                show_interval=0):
    """

    Args:
        model_path:         str con el path de la carpeta que guarda el modelo de TensorFlow que se usará para el test.
        class_names_path:   str con el path del archivo que guarda la lista de los nombres de las clases del modelo.
        test_dirs:          List[str] con los paths de los directorios con las imágenes que se usarán para sucesivos
                            tests.
        color_mode:         str con el tipo de color de las imágenes que se evalúan.
        show_interval:      int con la frecuencia con la que se muestran individualmente las imágenes testadas. Si es 0
                            no se muestra ninguna.
    """
    model = load_model(model_path)
    class_names = load(class_names_path)
    print(f'Testing model located in "{model_path}".')
    for test_dir in test_dirs:
        test_images(model, test_dir, class_names, color_mode=color_mode, show_interval=show_interval)
