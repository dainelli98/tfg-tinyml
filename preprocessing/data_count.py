import os
import os.path


def count_dir_files(directory: str) -> int:
    """
    Cuenta el número de archivos que contiene una carpeta.
    Args:
        directory:  str del directorio que contiene los archivos que se quiere contar.

    Returns:
        int que indica el número de archivos que contiene el directorio.
    """
    return len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])


def print_class_count(directory: str, data_desc=""):
    """
    Escribe el numero de muestras de cada clase en un directorio.
    Args:
        directory:  str del directorio que contiene las carpetas de las classes.
        data_desc:  str con una descripción opcional de los datos que se cuentan.
    """
    classes = os.listdir(directory)
    if data_desc != "":
        print(f"Datos: {data_desc}")
    for name in classes:
        count = count_dir_files(f"{directory}/{name}")
        print(f"\t{count} muestras de la clase {name}.")
