/**
 * Función que capta las imagenes con el módulo de camara del
 * microcontrolador y adapta su formato y las ubica en el input
 * tensor del modelo que se va usar para realizar inferencias.
 */

#ifndef IMAGE_PROVIDER_H_
#define IMAGE_PROVIDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include <Arduino_OV767X.h>

#define IMG_WIDTH 176
#define IMG_HEIGHT 144
#define INPUT_WIDTH 96
#define INPUT_HEIGHT 96
#define QUANT_FACTOR 0.9792773723602295
#define QUANT_OFFSET -128

/**
 * Inicializa el módulo de cámara de la placa del microcontrolador.
 * @param resolution: int que indica el tipo de resolución que usará la
 *                    cámara (VGA, CIF, QVGA, QCIF, QQVGA).
 * @param color:      int con el tipo de color que usará la cámara (YUV422,
 *                    RGB444, RGB565, GRAYSCALE).
 * @param fps:        int con el número de frames que capta la cámara por
 *                    segundo (1, 5, 10, 15, 30).
 */
void initialize_camera(int resolution, int color, int fps);

/**
 * Carga en el tensor input indicado los datos que capta la camara.
 * @param image data: Puntero al input tensor del interpreter que se quiere
 *                    usar para realizar inferencias.
 */
void prepare_image_data(int8_t* image_data);

#endif  // IMAGE_PROVIDER_H_
