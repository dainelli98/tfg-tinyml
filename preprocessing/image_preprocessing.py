from PIL import Image
import os


def centered_crop(filepath: str, width: int, height: int, output_path=False, return_image=False, im=False) -> Image:
    """
    Recorta los extremos de la imagen para que tenga la resolución indicada.
    Args:
        filepath:       str con el path del archivo que contiene la imagen que se quiere recortar.
        width:          int con la amplitud de la imagen que se quiere obtener como resultado en píxeles.
        height:         int con la altura de la imagen que se quiere obtener en píxeles.
        output_path:    str con el path del archivo que se quiere guardar con el resultado del recorte.
        return_image:   bool que indica que se quiere que se devuelva la imagen resultante de la operación.
        im:             Image previamente abierta, de uso alternativo a filepath.

    Returns:    Image recortada que se ha obtenido como resultado de la operación.

    """
    if not im:
        im = Image.open(filepath)

    initial_width, initial_height = im.size
    width_diff = initial_width - width
    height_diff = initial_height - height

    # Si se quiere aplicar un recorte mayor que el tamaño de la imagen original se interrumpe la operación.
    if width_diff < 0 or height_diff < 0:
        raise ValueError(f"Se quiere obtener una imagen de resolución {width}x{height} a partir de una imagen más"
                         f"pequeña {initial_width}x{initial_height} mediante recorte.")

    left = width_diff // 2
    right = initial_width - (width_diff - left)
    top = height_diff // 2
    bottom = initial_height - (height_diff - top)

    im = im.crop((left, top, right, bottom))

    if output_path:
        im.save(output_path)

    if return_image:
        return im


def to_grayscale(filepath: str, output_path=False, return_image=False, im=False) -> Image:
    """
    Convierte una imagen RGB a escala de grises.
    Args:
        filepath:       str con el path del archivo que contiene la imagen que se quiere recortar.
        output_path:    str con el path del archivo que se quiere guardar con el resultado del recorte.
        return_image:   bool que indica que se quiere que se devuelva la imagen resultante de la operación.
        im:             Image previamente abierta, de uso alternativo a filepath.

    Returns:    Image en escala de grises que se ha obtenido como resultado de la operación.

    """
    if not im:
        im = Image.open(filepath)

    im = im.convert('L')

    if output_path:
        im.save(output_path)

    if return_image:
        return im


def rescale(filepath: str, width: int, height: int, output_path=False, return_image=False, im=False) -> Image:
    """
    Reescala una imagen para que tenga la resolución indicada.
    Args:
        filepath:       str con el path del archivo que contiene la imagen que se quiere recortar.
        width:          int con la amplitud de la imagen que se quiere obtener como resultado en píxeles.
        height:         int con la altura de la imagen que se quiere obtener en píxeles.
        output_path:    str con el path del archivo que se quiere guardar con el resultado del recorte.
        return_image:   bool que indica que se quiere que se devuelva la imagen resultante de la operación.
        im:             Image previamente abierta, de uso alternativo a filepath.

    Returns:    Image reescalada que se ha obtenido como resultado de la operación.

    """
    if not im:
        im = Image.open(filepath)

    im = im.resize((width, height))

    if output_path:
        im.save(output_path)

    if return_image:
        return im


def micro_preprocessing(filepath, output_path=False, return_image=False, im=False) -> Image:
    """
    Aplica el preprocesado que se ha decidido aplicar a las imágenes captadas con microcontrolador.
    Args:
        filepath:       str con el path del archivo que contiene la imagen que se quiere recortar.
        output_path:    str con el path del archivo que se quiere guardar con el resultado del recorte.
        return_image:   bool que indica que se quiere que se devuelva la imagen resultante de la operación.
        im:             Image previamente abierta, de uso alternativo a filepath.

    Returns:    Image captada con microcontrolador preprocesada.

    """
    if not im:
        im = Image.open(filepath)

    im = to_grayscale("", return_image=True, im=im)
    im = centered_crop("", 96, 96, return_image=True, output_path=output_path, im=im)

    if return_image:
        return im
    
    
def external_mask_preprocessing(filepath, output_path=False, return_image=False, im=False) -> Image:
    """
    Aplica el preprocesado que se ha decidido aplicar a las imágenes externas de la clase mask.
    Args:
        filepath:       str con el path del archivo que contiene la imagen que se quiere recortar.
        output_path:    str con el path del archivo que se quiere guardar con el resultado del recorte.
        return_image:   bool que indica que se quiere que se devuelva la imagen resultante de la operación.
        im:             Image previamente abierta, de uso alternativo a filepath.

    Returns:    Image externa de la clase mask preprocesada.

    """
    if not im:
        im = Image.open(filepath)
        
    im = rescale("", 128, 128, return_image=True, im=im)
    im = to_grayscale("", return_image=True, im=im)
    im = centered_crop("", 96, 96, return_image=True, output_path=output_path, im=im)

    if return_image:
        return im


def external_nothing_preprocessing(filepath, output_path=False, return_image=False, im=False) -> Image:
    """
    Aplica el preprocesado que se ha decidido aplicar a las imágenes externas de la clase nothing.
    Args:
        filepath:       str con el path del archivo que contiene la imagen que se quiere recortar.
        output_path:    str con el path del archivo que se quiere guardar con el resultado del recorte.
        return_image:   bool que indica que se quiere que se devuelva la imagen resultante de la operación.
        im:             Image previamente abierta, de uso alternativo a filepath.

    Returns:    Image externa de la clase nothing preprocesada.

    """
    if not im:
        im = Image.open(filepath)

    width, height = im.size
    if width < height:
        new_width = 96
        new_height = max(96, int(height * (96 / width)))
    else:
        new_width = max(96, int(width * (96 / height)))
        new_height = 96

    im = rescale("", new_width, new_height, return_image=True, im=im)
    im = to_grayscale("", return_image=True, im=im)
    im = centered_crop("", 96, 96, output_path=output_path, return_image=True, im=im)

    if return_image:
        return im


def preprocess_external_mask_images(origin_path: str, destination_path: str):
    """
    Aplica el preprocesado para imágenes externas de la clase mask a todas las imágenes en origin_path y guarda el
    resultado en destination_path.
    Args:
        origin_path:        str con el path donde se encuentran las imágenes por preprocesar.
        destination_path:   str con el path donde se guardarán las imágenes preprocesadas.

    """
    # Creamos el directorio de destino si este no existe.
    try:
        os.makedirs(destination_path)
    except OSError as e:
        print("El directorio de destino ya existe.")

    img_id = 1
    for filename in os.listdir(origin_path):
        external_mask_preprocessing(f"{origin_path}/{filename}", output_path=f"{destination_path}/mask{img_id}.jpg")
        img_id += 1


def preprocess_external_nothing_images(origin_path: str, destination_path: str):
    """
    Aplica el preprocesado para imágenes externas de la clase nothing a todas las imágenes en origin_path y guarda el
    resultado en destination_path.
    Args:
        origin_path:        str con el path donde se encuentran las imágenes por preprocesar.
        destination_path:   str con el path donde se guardarán las imágenes preprocesadas.

    """
    # Creamos el directorio de destino si este no existe.
    try:
        os.makedirs(destination_path)
    except OSError as e:
        print("El directorio de destino ya existe.")

    img_id = 1
    for filename in os.listdir(origin_path):
        external_nothing_preprocessing(f"{origin_path}/{filename}", output_path=f"{destination_path}/mask{img_id}.jpg")
        img_id += 1


def preprocess_micro_images(origin_path: str, destination_path: str, classes=False):
    """
    Aplica el preprocesado para imágenes capturadas con el microcontrolador a todas las imágenes en origin_path
    (separadas en carpetas por clases) y guarda el resultado en destination_path. Por defecto se preprocesan todas las
    clases, pero se pueden delimitar con un listado.
    Args:
        origin_path:        str con el path donde se encuentran las imágenes por preprocesar.
        destination_path:   str con el path donde se guardarán las imágenes preprocesadas.
        classes:            lista de str con los nombres de las clases que se quieren preprocesar.

    """
    if not classes:
        classes = os.listdir(origin_path)

    for name in classes:
        origin = f"{origin_path}/{name}"
        destination = f"{destination_path}/{name}"

        # Creamos el directorio de destino si este no existe.
        try:
            os.makedirs(destination)
        except OSError as e:
            print("El directorio de destino ya existe.")

        img_id = 1
        for filename in os.listdir(origin):
            micro_preprocessing(f"{origin}/{filename}", output_path=f"{destination}/{name}{img_id}.jpg")
            img_id += 1


def preprocess_external_images(origin_path: str, destination_path: str, classes=False):
    """
    Aplica el preprocesado para imágenes externas a todas las imágenes en origin_path (separadas en carpetas por clases)
    y guarda el resultado en destination_path. Por defecto se preprocesan todas las clases, pero se pueden delimitar con
    un listado.
    Args:
        origin_path:        str con el path donde se encuentran las imágenes por preprocesar.
        destination_path:   str con el path donde se guardarán las imágenes preprocesadas.
        classes:            lista de str con los nombres de las clases que se quieren preprocesar.

    """
    if not classes:
        classes = os.listdir(origin_path)
    
    for name in classes:
        
        origin = f"{origin_path}/{name}"
        destination = f"{destination_path}/{name}"
            
        if name == "face":
            preprocess_micro_images(origin_path, destination_path, classes=["face"])
        
        elif name == "mask":
            preprocess_external_mask_images(origin, destination)
        else:
            preprocess_external_nothing_images(origin, destination)


if __name__ == '__main__':
    preprocess_micro_images("../samples/microcontroller/image", "../samples/microcontroller/preprocessed_image")
    preprocess_external_images("../samples/external/image", "../samples/external/preprocessed_image")
    pass

