from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from audio_processor import AudioProcessor
from typing import Dict, List, Any
import numpy as np
from six.moves import xrange


def window_count(sample_length: int, window_size: float, window_stride: int) -> int:
    """
    Calcula el número total de ventanas de audio que se pueden extraer de una muestra de audio.
    Args:
        sample_length:  int con la duración en milisegundos de una muestra.
        window_size:    int con la duración de las ventanas de audio.
        window_stride:  int con el desplazamiento que se produce entre el inicio de una ventana y el de la siguiente.

    Returns:
        int con el número total de ventanas de audio que se pueden extraer de una muestra de audio.
    """
    count = 0
    window_start = 0
    while window_start + window_size < sample_length:
        count += 1
        window_start += window_stride
    return count


def prepare_model_settings(sample_rate: int, sample_length: int, window_size: float, window_stride: int,
                           feature_bin_count: int) -> Dict[str, Any]:
    """
    Calcula los ajustes de un modelo.
    Args:
        sample_rate:        int con el ratio de muestreo que se deben tener las muestras.
        sample_length:      int con la longitud en milisegundos de las muestras de audio.
        window_size:        float con el tamaño en milisegundos de las ventanas de audio.
        window_stride:      int con el desplazamiento en milisegundos entre ventanas de audio.
        feature_bin_count:  int con el número de bins de frecuencia usados para análisis.

    Returns:
        Diccionario con los ajustes de un modelo.
    """
    desired_samples = int(sample_rate * sample_length / 1000)
    window_size_samples = int(sample_rate * window_size / 1000)
    window_stride_samples = int(sample_rate * window_stride / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    average_window_width = -1
    fingerprint_width = feature_bin_count
    fingerprint_size = fingerprint_width * spectrogram_length
    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "spectrogram_length": spectrogram_length,
        "fingerprint_width": fingerprint_width,
        "fingerprint_size": fingerprint_size,
        "average_window_width": average_window_width,
    }


def create_model(fingerprint_input, model_settings: Dict[str, Any], nclasses: int, tiny_model: bool):
    """
    Construye el modelo que se usará para el entrenamiento.
    Args:
        fingerprint_input:  Nodo de TensorFlow que contendrá los datos de entrada.
        model_settings:     Dict[str, Any] con los ajustes de un modelo.
        nclasses:           Numero de classes distintas que ha de tratar el modelo.
        tiny_model:         bool que indica si usar un modelo apto para TinyML o no.

    Returns:
        Nodo de TensorFlow que hace output de logits results y opcionalmente dropout placeholder.
    """
    if tiny_model:
        return create_tiny_conv_model(fingerprint_input, model_settings, nclasses)
    return create_conv_model(fingerprint_input, model_settings, nclasses)


def create_conv_model(fingerprint_input, model_settings: Dict[str, Any], nclasses: int):
    """
    Construye un modelo Convolucional estándar no apto para TinyML pero útil en comparativas
    Args:
        fingerprint_input:  Nodo de TensorFlow que contendrá los datos de entrada.
        model_settings:     Dict[str, Any] con los ajustes de un modelo.
        nclasses:           Numero de classes distintas que ha de tratar el modelo.

    Returns:
        Nodo de TensorFlow que hace output de logits results y opcionalmente dropout placeholder.
    """
    print("Usando modelo Conv estándar.")
    dropout_rate = tf.compat.v1.placeholder(tf.float32, name="dropout_rate")
    input_frequency_size = model_settings["fingerprint_width"]
    input_time_size = model_settings["spectrogram_length"]
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = 20
    first_filter_count = 64
    first_weights = tf.compat.v1.get_variable(name="first_weights",
                                              initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
                                              shape=[first_filter_height, first_filter_width, 1, first_filter_count])
    first_bias = tf.compat.v1.get_variable(name="first_bias", initializer=tf.compat.v1.zeros_initializer,
                                           shape=[first_filter_count])
    first_conv = (tf.nn.conv2d(input=fingerprint_4d, filters=first_weights, strides=[1, 1, 1, 1], padding="SAME")
                  + first_bias)
    first_relu = tf.nn.relu(first_conv)
    first_dropout = tf.nn.dropout(first_relu, rate=dropout_rate)
    max_pool = tf.nn.max_pool2d(input=first_dropout, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    second_filter_width = 4
    second_filter_height = 10
    second_filter_count = 64
    second_weights = tf.compat.v1.get_variable(name="second_weights",
                                               initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
                                               shape=[second_filter_height, second_filter_width, first_filter_count,
                                                      second_filter_count])
    second_bias = tf.compat.v1.get_variable(name="second_bias", initializer=tf.compat.v1.zeros_initializer,
                                            shape=[second_filter_count])
    second_conv = (tf.nn.conv2d(input=max_pool, filters=second_weights, strides=[1, 1, 1, 1], padding="SAME")
                   + second_bias)
    second_relu = tf.nn.relu(second_conv)
    second_dropout = tf.nn.dropout(second_relu, rate=dropout_rate)
    second_conv_shape = second_dropout.get_shape()
    second_conv_output_width = second_conv_shape[2]
    second_conv_output_height = second_conv_shape[1]
    second_conv_element_count = int(second_conv_output_width * second_conv_output_height * second_filter_count)
    flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count])
    final_fc_weights = tf.compat.v1.get_variable(name="final_fc_weights",
                                                 initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
                                                 shape=[second_conv_element_count, nclasses])
    final_fc_bias = tf.compat.v1.get_variable(name="final_fc_bias", initializer=tf.compat.v1.zeros_initializer,
                                              shape=[nclasses])
    final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
    return final_fc, dropout_rate


def create_tiny_conv_model(fingerprint_input, model_settings: Dict[str, Any], nclasses: int):
    """
    Construye un modelo Tiny Conv.
    Args:
        fingerprint_input:  Nodo de TensorFlow que contendrá los datos de entrada.
        model_settings:     Dict[str, Any] con los ajustes de un modelo.
        nclasses:           Numero de classes distintas que ha de tratar el modelo.

    Returns:
        Nodo de TensorFlow que hace output de logits results y opcionalmente dropout placeholder.
    """
    print("Usando modelo TinyConv.")
    dropout_rate = tf.compat.v1.placeholder(tf.float32, name="dropout_rate")
    input_frequency_size = model_settings["fingerprint_width"]
    input_time_size = model_settings["spectrogram_length"]
    fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = 10
    first_filter_count = 8
    first_weights = tf.compat.v1.get_variable(name="first_weights",
                                              initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
                                              shape=[first_filter_height, first_filter_width, 1, first_filter_count])
    first_bias = tf.compat.v1.get_variable(name="first_bias", initializer=tf.compat.v1.zeros_initializer,
                                           shape=[first_filter_count])
    first_conv_stride_x = 2
    first_conv_stride_y = 2
    first_conv = tf.nn.conv2d(input=fingerprint_4d, filters=first_weights,
                              strides=[1, first_conv_stride_y, first_conv_stride_x, 1], padding="SAME") + first_bias
    first_relu = tf.nn.relu(first_conv)
    first_dropout = tf.nn.dropout(first_relu, rate=dropout_rate)
    first_dropout_shape = first_dropout.get_shape()
    first_dropout_output_width = first_dropout_shape[2]
    first_dropout_output_height = first_dropout_shape[1]
    first_dropout_element_count = int(first_dropout_output_width * first_dropout_output_height * first_filter_count)
    flattened_first_dropout = tf.reshape(first_dropout, [-1, first_dropout_element_count])
    final_fc_weights = tf.compat.v1.get_variable(name="final_fc_weights",
                                                 initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
                                                 shape=[first_dropout_element_count, nclasses])
    final_fc_bias = tf.compat.v1.get_variable(name="final_fc_bias", initializer=tf.compat.v1.zeros_initializer,
                                              shape=[nclasses])
    final_fc = (tf.matmul(flattened_first_dropout, final_fc_weights) + final_fc_bias)
    return final_fc, dropout_rate


def debug_get_sample_dist(train_ground_truth: List[float], class_index: Dict[str, int]):
    """
    Función debug para comprobar que existe balance en las etiquetas de las muestras que obtiene get_data.
    Args:
        train_ground_truth: np.Array[float] con las etiquetas de las muestras.
        class_index:        Dict[str, float] que mapea clases con el índice que las representa.

    Returns:

    """
    sumlist = np.zeros(len(class_index))
    for label in train_ground_truth:
        sumlist[int(label)] += 1
    print("\nOcurrencia de clases en la muestra:")
    for name in class_index:
        print(f"\t-{name}:\t{int(sumlist[class_index[name]])}")


def train_model(data_dir: str, test_dir: str, train_dir: str, logs_dir: str, label_index_dir: str, classes: str,
                sample_rate: int, sample_length: int, window_size: float, window_stride: int, time_shift: float,
                feature_bin_count: int, tiny_model: bool, training_steps: List[int], learning_rates: List[float],
                validation_percentage: float, test_percentage: float, batch_size: int, seed: int, model_name: str):
    """
    Entrena un modelo de TensorFlow de KeyWord Spotting.
    Args:
        data_dir:               str con el path donde se almacenan los datos de sonido.
        test_dir:               str con el path donde se almacena el archivo con la lista de muestras asignadas a test.
        train_dir:              str con el path donde se almacenan los datos de entrenamiento.
        logs_dir:               str con el path donde se almacena los logs.
        label_index_dir:        str con el path donde se almacena los archivos con los indices de las labels.
        classes:                str list con los nombres de las clases de los datos de sonido.
        sample_rate:            int con el ratio de muestreo que se deben tener las muestras.
        sample_length:          int con la longitud en milisegundos de las muestras de audio.
        window_size:            float con el tamaño en milisegundos de las ventanas de audio.
        window_stride:          int con el desplazamiento en milisegundos entre ventanas de audio.
        time_shift:             float que indica el máximo desplazamiento aplicable a las muestras.
        feature_bin_count:      int con el número de bins de frecuencia usados para análisis.
        tiny_model:             bool que indica si usar un modelo apto para TinyML o no.
        training_steps:         int list con los números de pasos de entrenamiento.
        learning_rates:         float list con los learning rates de entrenamiento.
        validation_percentage:  float con el porcentaje de las muestras que se usa para validación.
        test_percentage:        float con el porcentaje de las muestras que se usa para testeo.
        batch_size:             int con el número de muestras que se toman por iteración de training loop.
        seed:                   int con la seed que se quiere usar.
        model_name:             str con el nombre del modelo que se entrenará.
    """

    # Indicamos ejecución verbose para tener información durante la ejecución.
    tf.compat.v1.logging.set_verbosity("DEBUG")
    # Iniciamos una sesión interactiva de TensorFlow.
    sess = tf.compat.v1.InteractiveSession()

    # Se calculan algunos parámetros adicionales del modelo.
    model_settings = prepare_model_settings(sample_rate, sample_length, window_size, window_stride, feature_bin_count)
    fingerprint_size = model_settings["fingerprint_size"]
    label_count = len(classes)
    time_shift_samples = int((time_shift * sample_rate) / 1000)

    # Se inicializa un procesador de muestras de audio.
    audio_processor = AudioProcessor(data_dir, test_dir, logs_dir, label_index_dir, classes, sample_rate,
                                     validation_percentage, test_percentage, model_settings, seed, model_name)

    # Se prepara input placeholder y se ajusta si se usa cuantización o no en el entrenamiento.
    input_placeholder = tf.compat.v1.placeholder(tf.float32, [None, fingerprint_size], name="fingerprint_input")
    fingerprint_input = input_placeholder

    # Se crea el modelo con la arquitectura deseada.
    logits, dropout_rate = create_model(fingerprint_input, model_settings, len(classes), tiny_model)

    # Definimos la función de pérdida y el optimizador de la red.
    ground_truth_input = tf.compat.v1.placeholder(tf.int64, [None], name="groundtruth_input")

    # Añadimos comprobaciones para detectar errores con NaNs durante el entrenamiento.
    checks = tf.compat.v1.add_check_numerics_ops()
    control_dependencies = [checks]

    # Configuramos back propagation y sistema de evaluación del entrenamiento.
    with tf.compat.v1.name_scope("cross_entropy"):
        cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=logits)
    with tf.compat.v1.name_scope("train"), tf.control_dependencies(control_dependencies):
        learning_rate_input = tf.compat.v1.placeholder(tf.float32, [], name="learning_rate_input")
        train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate_input).minimize(cross_entropy_mean)
    predicted_indices = tf.argmax(input=logits, axis=1)
    correct_prediction = tf.equal(predicted_indices, ground_truth_input)
    confusion_matrix = tf.math.confusion_matrix(labels=ground_truth_input, predictions=predicted_indices,
                                                num_classes=label_count)
    evaluation_step = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
    with tf.compat.v1.get_default_graph().name_scope("eval"):
        tf.compat.v1.summary.scalar("cross_entropy", cross_entropy_mean)
        tf.compat.v1.summary.scalar("accuracy", evaluation_step)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    # Hacemos log de lo que se ha realizado hasta ahora
    merged_summaries = tf.compat.v1.summary.merge_all(scope="eval")
    train_writer = tf.compat.v1.summary.FileWriter(logs_dir + "/train", sess.graph)
    validation_writer = tf.compat.v1.summary.FileWriter(logs_dir + "/validation")

    tf.compat.v1.global_variables_initializer().run()
    start_step = 1

    # Checkpoint al inicio
    tf.compat.v1.logging.info("Training from step: %d ", start_step)
    model_ini_filename = model_name + ".pbtxt"    # Python 2.7 no soporta fstring.
    tf.io.write_graph(sess.graph_def, train_dir, model_ini_filename)

    # Training loop
    training_steps_max = np.sum(training_steps)
    for training_step in xrange(start_step, training_steps_max + 1):
        print(f"Training step: {training_step}")

        training_steps_sum = 0

        # Se comprueba si se debe cambiar el learning rate.
        for i in range(len(training_steps)):
            training_steps_sum += training_steps[i]
            if training_step <= training_steps_sum:
                learning_rate_value = learning_rates[i]
                break
        # Cargamos las muestras de audio que se usarán para el entrenamiento.
        train_fingerprints, train_ground_truth = audio_processor.get_data(batch_size, 0, model_settings,
                                                                          time_shift_samples, "training", sess)

        # Debug para comprobar que la selección no determinística no afecta a la distribución de clases de las muestras
        # recogidas.
        # debug_get_sample_dist(train_ground_truth, audio_processor.class_index.copy())

        # Se ejecuta una iteración del entrenamiento con los datos de muestra.
        train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
            [
                merged_summaries,
                evaluation_step,
                cross_entropy_mean,
                train_step,
                increment_global_step
            ],
            feed_dict={
                fingerprint_input: train_fingerprints,
                ground_truth_input: train_ground_truth,
                learning_rate_input: learning_rate_value,
                dropout_rate: 0.5
            }
        )

        # Log del resultado del entrenamiento.
        train_writer.add_summary(train_summary, training_step)
        tf.compat.v1.logging.debug("Step #%d: rate %f, accuracy %.1f%%, cross entropy %f" % (training_step,
                                                                                             learning_rate_value,
                                                                                             train_accuracy * 100,
                                                                                             cross_entropy_value))

        # Se comprueba si se ha llegado a la última iteración.
        last_step = training_step == training_steps_max

        # Cada 1000 iteraciones y en la iteración final se prueba el rendimiento del modelo con la partición de
        # validación y se estudia el rendimiento. También se guarda el modelo.
        if (training_step % 1000) == 0 or last_step:
            tf.compat.v1.logging.info("Step #%d: rate %f, accuracy %.1f%%, cross entropy %f" % (training_step,
                                                                                                learning_rate_value,
                                                                                                train_accuracy * 100,
                                                                                                cross_entropy_value))
            partition_size = audio_processor.get_partition_size("validation")
            total_accuracy = 0
            total_conf_matrix = None
            for i in xrange(0, partition_size, batch_size):
                validation_fingerprints, validation_ground_truth = (audio_processor.get_data(batch_size, i,
                                                                                             model_settings, 0,
                                                                                             "validation", sess))
                validation_summary, validation_accuracy, conf_matrix = sess.run(
                    [
                        merged_summaries,
                        evaluation_step,
                        confusion_matrix],
                    feed_dict={
                        fingerprint_input: validation_fingerprints,
                        ground_truth_input: validation_ground_truth,
                        dropout_rate: 0.0
                    })
                validation_writer.add_summary(validation_summary, training_step)
                batch_size = min(batch_size, partition_size - i)
                total_accuracy += (validation_accuracy * batch_size) / partition_size
                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    total_conf_matrix += conf_matrix
            tf.compat.v1.logging.info("Confusion Matrix:\n %s" % total_conf_matrix)
            tf.compat.v1.logging.info("Step %d: Validation accuracy = %.1f%% (N=%d)" % (training_step,
                                                                                        total_accuracy * 100,
                                                                                        partition_size))

            # Guardamos checkpoint del modelo.
            checkpoint_path = f"{train_dir}/model/{model_name}.ckpt"
            tf.compat.v1.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)

    # Al acabar de entrenar el modelo se ejecuta la partición test.
    partition_size = audio_processor.get_partition_size("testing")
    tf.compat.v1.logging.info('set_size=%d', partition_size)
    total_accuracy = 0
    total_conf_matrix = None
    for i in xrange(0, partition_size, batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(batch_size, i, model_settings, 0, 'testing',
                                                                        sess)
        test_accuracy, conf_matrix = sess.run(
            [
                evaluation_step,
                confusion_matrix
            ],
            feed_dict={
                fingerprint_input: test_fingerprints,
                ground_truth_input: test_ground_truth,
                dropout_rate: 0.0
            })
        batch_size = min(batch_size, partition_size - i)
        total_accuracy += (test_accuracy * batch_size) / partition_size
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.compat.v1.logging.warn('Confusion Matrix:\n %s' % total_conf_matrix)
    tf.compat.v1.logging.warn('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100, partition_size))
