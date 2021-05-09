import os
import random as rnd
from shutil import copyfile


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

    if os.path.isdir(f"{data_dir}/train"):
        os.rmdir(f"{data_dir}/train")
    
    if os.path.isdir(f"{data_dir}/test"):
        os.rmdir(f"{data_dir}/test")

    classes = os.listdir(data_dir)
    os.mkdir(f"{data_dir}/train")
    os.mkdir(f"{data_dir}/test")

    print(f"\nParticionando datos:\n\t- Directorio: {data_dir}\n\t- Porcetage test: {test_percentage}")

    for name in classes:

        if name not in ["train", "test", "all"]:

            if not os.path.isdir(f"{data_dir}/train/{name}"):
                os.mkdir(f"{data_dir}/train/{name}")
            if not os.path.isdir(f"{data_dir}/test/{name}"):
                os.mkdir(f"{data_dir}/test/{name}")

            train_count = 0
            test_count = 0
            files = os.listdir(f"{data_dir}/{name}")
            rnd.shuffle(files)
            nfiles = len(files)
            test_samples = nfiles * (test_percentage / 100.)
            train_samples = nfiles - test_samples

            for file in files:
                
                prob = rnd.random()

                if test_count >= test_samples or (train_count < train_samples and prob > (test_percentage / 100.)):
                    copyfile(f"{data_dir}/{name}/{file}", f"{data_dir}/train/{name}/{name}{train_count + 1}.{extension}")
                    train_count += 1
                else:
                    copyfile(f"{data_dir}/{name}/{file}", f"{data_dir}/test/{name}/{name}{test_count + 1}.{extension}")
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
