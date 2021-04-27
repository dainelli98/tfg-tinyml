import random as rnd
import os
from typing import Dict, List, Any, Union
import joblib as jbl
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.summary.writer.writer import FileWriter
import numpy as np
from six.moves import xrange
try:
    from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
except ImportError:
    frontend_op = None


tf.compat.v1.disable_eager_execution()


def get_random_elements_from_list(element_list: [Any], nelements: int) -> (List[Any], List[Any]):
    """
    Sustrae un conjunto aleatorio de elementos de una lista y devuelve los dos conjuntos resultantes.
    Args:
        element_list:   List[Any] de la que se sustraen los elementos.
        nelements:      int que indica el número de elementos que se quiere sustraer.

    Returns:
        (List[Any], List[Any]) con los elementos sustraídos y el resto de elementos.
    """
    chosen = rnd.sample(element_list, nelements)
    rest = [element for element in element_list if element not in chosen]
    return chosen, rest


def partition_paths(file_paths: [str], nvalidation_samples: int, ntest_samples: int) -> (List[str], List[str],
                                                                                         List[str]):
    """
    Divide un listado de file_paths en tres listados con los paths relativos a las particiones validation, test, y
    train.
    Args:
        file_paths:             List[str] con los file_paths que se quieren particionar.
        nvalidation_samples:   int con el número de elementos que debe tener la partición validation.
        ntest_samples:         int con el número de elementos que debe tener la partición test.

    Returns:
        (List[str], List[str], List[str]) con las particiones validation, test y train.
    """
    validation_paths, rest = get_random_elements_from_list(file_paths, nvalidation_samples)
    test_paths, train_paths = get_random_elements_from_list(rest, ntest_samples)
    return validation_paths, test_paths, train_paths


class AudioProcessor:
    """
    Gestiona la partición y preparación de los datos de entrenamiento de sonido.
    """
    class_index: Dict[Any, int]
    summary_writer_: FileWriter
    merged_summaries_: Union[Union[None, object, Tensor], Any]
    output_: Union[object, Any]
    background_volume_placeholder_: Union[object, Any]
    background_data_placeholder_: Union[object, Any]
    background_data: List[Any]
    time_shift_offset_placeholder_: Union[object, Any]
    time_shift_padding_placeholder_: Union[object, Any]
    foreground_volume_placeholder_: Union[object, Any]
    wav_filename_placeholder_: Union[object, Any]
    data_index: Dict[str, List[Any]]

    def __init__(self, data_dir: str, test_dir: str, logs_dir: str, label_index_dir: str, classes: [str],
                 sample_rate: int, validation_percentage: float, test_percentage: float, model_settings: Dict[str, Any],
                 seed: int, model_name: str):
        """
        Constructora de la clase AudioProcessor.
        Args:
            data_dir:               str con el path donde se almacenan los datos de sonido.
            test_dir:               str con el path donde se almacena el archivo con la lista de muestras asignadas a
                                    test.
            logs_dir:               str con el path donde se almacena los logs.
            label_index_dir:        str con el path donde se almacena los archivos con los indices de las labels.
            classes:                str list con los nombres de las clases de los datos de sonido.
            sample_rate:            int con el ratio de muestreo que se deben tener las muestras.
            validation_percentage:  float con el porcentaje de las muestras que se usa para validación.
            test_percentage:        float con el porcentaje de las muestras que se usa para testeo.
            model_settings:         Dict[str, Any] con los ajustes de un modelo.
            seed:                   int con la seed que se quiere usar.
            model_name:             str con el nombre del modelo que se entrenará.
        """
        self.data_dir = data_dir
        self.logs_dir = logs_dir
        self.seed = seed
        self.classes = classes
        self.sample_rate = sample_rate
        self.prepare_data_partitions(validation_percentage, test_percentage, test_dir, label_index_dir, model_name)
        self.prepare_processing_protocol(model_settings)

    def prepare_data_partitions(self, validation_percentage: float, test_percentage: float,
                                test_dir: str, label_index_dir: str, model_name: str):
        """
        Organiza las particiones de las muestras y guarda un archivo con el listado de muestras para test. Se mantiene
        la distribución original de clases.
        Args:
            validation_percentage:  float con el porcentaje de las muestras que se usa para validación.
            test_percentage:        float con el porcentaje de las muestras que se usa para testeo.
            test_dir:               str con el path donde se almacena el archivo con la lista de muestras asignadas a
                                    test.
            label_index_dir:        str con el path donde se almacena los archivos con los indices de las labels.
            model_name:             str con el nombre del modelo que se entrenará.
        """
        rnd.seed(self.seed)
        self.class_index = {}
        index_label_txt = ""
        for index, name in enumerate(self.classes):
            self.class_index[name] = index
            index_label_txt += f"{index}: {name}\n"

        # Guardamos los indices de las labels para poder disponer de esta información en el futuro.
        with open(f"{label_index_dir}/{model_name}.txt", 'wt') as label_file:
            label_file.write(index_label_txt)

        self.data_index = {"validation": [], "testing": [], "training": []}
        for name in self.classes:
            class_dir = f"{self.data_dir}/{name}"
            files = os.listdir(class_dir)
            nvalidation_samples = int(len(files) * (validation_percentage / 100))
            ntest_samples = int(len(files) * (test_percentage / 100))
            file_paths = [f"{class_dir}/{file}" for file in files]
            validation_paths, test_paths, train_paths = partition_paths(file_paths, nvalidation_samples, ntest_samples)
            for path in validation_paths:
                self.data_index["validation"].append({'label': name, 'file': path})
            for path in test_paths:
                self.data_index["testing"].append({'label': name, 'file': path})
            for path in train_paths:
                self.data_index["training"].append({'label': name, 'file': path})
            jbl.dump(self.data_index["testing"], f"{test_dir}/{model_name}.joblib")
        for partition in ['validation', 'testing', 'training']:
            rnd.shuffle(self.data_index[partition])

    def prepare_processing_protocol(self, model_settings: Dict[str, Any]):
        """
        Construye un TensorFlow graph que aplica el preprocesado a las muestras de audio
        Args:
            model_settings: Dict[str, Any] que contiene la configuración del modelo.
        """
        with tf.compat.v1.get_default_graph().name_scope('data'):
            desired_samples = model_settings["desired_samples"]
            self.wav_filename_placeholder_ = tf.compat.v1.placeholder(tf.string, [], name='wav_filename')
            wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
            wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)

            # Ajuste de volumen.
            self.foreground_volume_placeholder_ = tf.compat.v1.placeholder(tf.float32, [], name='foreground_volume')
            scaled_foreground = tf.multiply(wav_decoder.audio, self.foreground_volume_placeholder_)

            # Desplazamiento del inicio de la muestra.
            self.time_shift_padding_placeholder_ = tf.compat.v1.placeholder(tf.int32, [2, 2], name='time_shift_padding')
            self.time_shift_offset_placeholder_ = tf.compat.v1.placeholder(tf.int32, [2], name='time_shift_offset')
            padded_foreground = tf.pad(tensor=scaled_foreground, paddings=self.time_shift_padding_placeholder_,
                                       mode='CONSTANT')
            sliced_foreground = tf.slice(padded_foreground, self.time_shift_offset_placeholder_, [desired_samples, -1])

            # Se crea placeholder para background data
            self.background_data = []
            # print(f"desired_samples: {desired_samples}")
            self.background_data_placeholder_ = tf.compat.v1.placeholder(tf.float32, [desired_samples, 1],
                                                                         name='background_data')
            self.background_volume_placeholder_ = tf.compat.v1.placeholder(tf.float32, [], name='background_volume')
            background_mul = tf.multiply(self.background_data_placeholder_, self.background_volume_placeholder_)
            background_add = tf.add(background_mul, sliced_foreground)
            background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)

            # Computo de espectrograma y MFCC.
            spectrogram = audio_ops.audio_spectrogram(background_clamp,
                                                      window_size=model_settings['window_size_samples'],
                                                      stride=model_settings['window_stride_samples'],
                                                      magnitude_squared=True)
            tf.compat.v1.summary.image('spectrogram', tf.expand_dims(spectrogram, -1), max_outputs=1)
            if not frontend_op:
                raise Exception("Micro frontend op is currently not available when running TensorFlow directly from"
                                " Python, you need to build and run through Bazel")
            else:
                print("Frontend disponible.")
            window_size_ms = (model_settings['window_size_samples'] * 1000) / self.sample_rate
            window_step_ms = (model_settings['window_stride_samples'] * 1000) / self.sample_rate
            int16_input = tf.cast(tf.multiply(background_clamp, 32768), tf.int16)
            micro_frontend = frontend_op.audio_microfrontend(int16_input, sample_rate=self.sample_rate,
                                                             window_size=window_size_ms, window_step=window_step_ms,
                                                             num_channels=model_settings['fingerprint_width'],
                                                             out_scale=1, out_type=tf.float32)
            self.output_ = tf.multiply(micro_frontend, (10.0 / 256.0))
            tf.compat.v1.summary.image('micro', tf.expand_dims(tf.expand_dims(self.output_, -1), 0), max_outputs=1)
            self.merged_summaries_ = tf.compat.v1.summary.merge_all(scope='data')
            self.summary_writer_ = tf.compat.v1.summary.FileWriter(self.logs_dir + '/data',
                                                                   tf.compat.v1.get_default_graph())

    def get_data(self, nsamples: int, offset: int, model_settings: Dict[str, Any], time_shift: float, mode: str,
                 session) -> (List[List[float]], List[float]):
        """
        Recoge un conjunto de muestras de la partición indicada. Aplicando el preprocesado necesario. En el caso de la
        partición training la muestra se coge de forma aleatoria.
        Args:
            nsamples:       int con el número de muestras que se extraen.
            offset:         int con el punto de inicio al extraer de forma determinística.
            model_settings: Dict[str, Any] con los ajustes de un modelo.
            time_shift:     float que indica el máximo desplazamiento aplicable a las muestras.
            mode:           str que indica la partición de la que se quiere extraer las muestras.
            session:        TensorFlow session activa cuando se creo el AudioProcessor.

        Returns:
            (List[[float]]], List[float]) con los datos de muestra y sus labels.
        """
        # Seleccionamos partición y se establece número de muestras definitivo.
        partition = self.data_index[mode]
        nsamples = max(0, min(nsamples, len(partition) - offset))

        # Inicializamos arrays data y labels.
        data = np.zeros((nsamples, model_settings['fingerprint_size']))
        labels = np.zeros(nsamples)

        desired_samples = model_settings['desired_samples']
        deterministic = (mode != 'training')

        # Se utiliza el processing graph creado para preprocesar las muestras de audio.
        for i in xrange(offset, offset + nsamples):
            if deterministic:
                sample_index = i
            else:
                sample_index = np.random.randint(len(partition))

            sample = partition[sample_index]

            # Se aplica time_shift random
            if time_shift > 0:
                sample_time_shift = np.random.randint(-time_shift, time_shift)
            else:
                sample_time_shift = 0
            if sample_time_shift > 0:
                time_shift_padding = [[sample_time_shift, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -sample_time_shift], [0, 0]]
                time_shift_offset = [-sample_time_shift, 0]

            # Guardamos los datos de input en un diccionario. No se añade sonido de fondo dado que se considera que las
            # muestras ya contienen.
            input_dict = {
                self.wav_filename_placeholder_: sample['file'],
                self.time_shift_padding_placeholder_: time_shift_padding,
                self.time_shift_offset_placeholder_: time_shift_offset,
                self.background_data_placeholder_: np.zeros([desired_samples, 1], dtype=float),
                self.background_volume_placeholder_: 0.0,
                self.foreground_volume_placeholder_: 1.0,
            }

            # Se ejecuta el graph que aplica el preprocesado
            summary, data_tensor = session.run([self.merged_summaries_, self.output_], feed_dict=input_dict)
            self.summary_writer_.add_summary(summary)
            data[i - offset, :] = data_tensor.flatten()
            label_index = self.class_index[sample['label']]
            labels[i - offset] = label_index

        # Devolvemos los arrays de datos y labels
        return data, labels

    def get_partition_size(self, partition_name: str) -> int:
        """
        Devuelve el número de muestras que tiene una partición.
        Args:
            partition_name: str con el nombre de la partición de la que se quiere saber el tamaño.

        Returns:
            int con el número de muestras que tiene la partición indicada.
        """
        return len(self.data_index[partition_name])
