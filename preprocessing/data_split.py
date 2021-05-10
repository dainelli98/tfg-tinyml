import os
import random as rnd
from shutil import copyfile
from typing import List

# Constantes
TRAIN: bool = True
TEST: bool = False


def clean_existing_partition(data_dir: str):
    """
    Si la hay, elimina particiones previamente existentes donde se va a guardar la nueva.
    Args:
        data_dir:   str con el path de la carpeta que puede contener una partición anterior.
    """
    if os.path.isdir(f"{data_dir}/train"):
        os.rmdir(f"{data_dir}/train")

    if os.path.isdir(f"{data_dir}/test"):
        os.rmdir(f"{data_dir}/test")


def create_output_directories(data_dir: str, class_names: List[str]):
    """
    Crea los directorios donde se guardarán los datos de las particiones.
    Args:
        data_dir:       str con el path de la carpeta que puede contener una partición anterior.
        class_names:    List[str] que contiene los nombres de las clases de los datos que se quiere particionar.
    """
    os.mkdir(f"{data_dir}/train")
    os.mkdir(f"{data_dir}/test")
    for name in class_names:
        if name not in ["train", "test", "all"]:
            os.mkdir(f"{data_dir}/train/{name}")
            os.mkdir(f"{data_dir}/test/{name}")


def prepare_partition_directories(data_dir: str, class_names):
    """
    Prepara los directorios que se usarán para guardar una nueva partición de datos.
    Args:
        data_dir:       str con el path de la carpeta que puede contener una partición anterior.
        class_names:    List[str] que contiene los nombres de las clases de los datos que se quiere particionar.
    """
    clean_existing_partition(data_dir)
    create_output_directories(data_dir, class_names)
    

def assign_partition(filename: str, data_dir: str, class_name: str, extension: str, test_percentage: float, train_count=0,
                     test_count=0) -> bool:
    """
    Asigna una muestra a una partición de forma aleatoria. La copia en la carpeta correspondiente.
    Args:
        filename:           str con el nombre del archivo (no se debe usar path, solo nombre).
        data_dir:           str con el path de la carpeta que puede contener una partición anterior.
        class_name:         str con el nombre de la clase de la muestra que se asigna.
        extension:          str con la extensión de los archivos de los datos que se particionan.
        test_percentage:    float con el porcentaje de muestras que deben destinarse a test.
        train_count:        int con el número de muestras ya asignadas a la partición train.
        test_count:         int con el número de muestras ya asignadas a la partición test.

    Returns:
        bool TRAIN si la muestra se asigna a la partición de entrenamiento o TEST en caso contrario.
    """
    prob = rnd.random()

    if test_count >= test_samples or (train_count < train_samples and prob > (test_percentage / 100.)):
        copyfile(f"{data_dir}/{class_name}/{filename}", f"{data_dir}/train/{class_name}/{class_name}{train_count + 1}.{extension}")
        train_count += 1
        return TRAIN

    copyfile(f"{data_dir}/{class_name}/{filename}", f"{data_dir}/test/{class_name}/{class_name}{test_count + 1}.{extension}")
    test_count += 1
    return TEST


def partition_data(data_dir: str, test_percentage: float, extension: str, seed: int):
    """
    Particiona un conjunto de datos separados en carpetas por clase en conjuntos de entrenamiento y validación.
    Args:
        data_dir:           str con el path del directorio donde se encuentran los datos.
        test_percentage:    float con el porcentaje de muestras que deben destinarse a test.
        extension:          str con la extensión de los archivos de los datos que se particionan.
        seed:               int con una seed para hace la función determinística.
    """
    rnd.seed(seed)

    class_names = os.listdir(data_dir)

    prepare_partition_directories(data_dir, class_names)

    print(f"\nParticionando datos:\n\t- Directorio: {data_dir}\n\t- Porcetage test: {test_percentage}")

    for name in class_names:

        if name not in ["train", "test", "all"]:

            train_count = 0
            test_count = 0
            files = os.listdir(f"{data_dir}/{name}")
            rnd.shuffle(files)
            nfiles = len(files)
            test_samples = nfiles * (test_percentage / 100.)
            train_samples = nfiles - test_samples

            for file in files:
                
                assignment = assign_partition(data_dir, name, extension, test_percentage, train_count=train_count,
                                              test_count=test_count)

                prob = rnd.random()

                if assignment = TRAIN
                    train_count += 1
                else:
                    test_count += 1

            print(f"\t- {nfiles} muestras de la clase {name} donde {train_count} son para entrenamiento y {test_count}"
                  f" son para test.")


if __name__ == '__main__':
    seed = 3244

    partition_data("/home/daniel/PycharmProjects/tfg-tinyml/samples/external/audio", 20, "wav", seed)
    partition_data("/home/daniel/PycharmProjects/tfg-tinyml/samples/external/preprocessed image", 20, "jpg", seed)
    partition_data("/home/daniel/PycharmProjects/tfg-tinyml/samples/microcontroller/audio", 20, "wav", seed)
    partition_data("/home/daniel/PycharmProjects/tfg-tinyml/samples/microcontroller/preprocessed image", 20, "jpg",
                   seed)
