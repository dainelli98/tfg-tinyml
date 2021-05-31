/**
 * Funciones dedicadas a la interpretación de los resultados
 * de una inferencia sobre datos de imagen y actuar correspondiente
 * a esa inferencia.
==============================================================================*/

#ifndef IMAGE_RESPONDER_H_
#define IMAGE_RESPONDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "Arduino.h"

/**
 * Inicializa los leds que usa el responder.
 */
void initialize_responder();

/**
 * Interpreta los resultados de una inferencia y actua en consecuencia.
 * @param error_reporter: error reporter de TFLite que se está usando.
 * @param face_score:     int8 con la puntuación obtenida para la clase face.
 * @param mask_score:     int8 con la puntuación obtenida para la clase mask.
 * @param nothing_score:  int8 con la puntuación obtenida para la clase nothing.
 */
void respond_image_inference(tflite::ErrorReporter* error_reporter,
                             int8_t face_score, int8_t mask_score,
                             int8_t nothing_score);

#endif  // IMAGE_RESPONDER_H_
