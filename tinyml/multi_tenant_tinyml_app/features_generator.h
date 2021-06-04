/*
 * Funciones empleadas para generar FFT a partir de los datos de audio
 * registrados con el micrófono.
==============================================================================*/

#ifndef FEATURES_GENERATOR_H_
#define FEATURES_GENERATOR_H_

#include <cmath>
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

#include "audio_model_settings.h"

#define FIXED_POINT 16
#define QUANT_FACTOR 0.3137255311012268
#define QUANT_OFFSET 127
/*
 * Prepara el feature generator.
 * @param error_reporter: error reporter usado en la aplicación.
 */
TfLiteStatus initialize_features(tflite::ErrorReporter* error_reporter);

/*
 * Prepara en output el resultado de aplicar FFT a los datos de audio
 * recibidos en input.
 * @param error_reporter:   error reporter usado en la aplicación.
 * @param input:            Puntero a los datos de audio sin procesar.
 * @param input_size:       Longitud de los datos de audio sin procesar.
 * @param output_size:      Longitud de los datos procesados de salida.
 * @param output:           Puntero a los datos procesados de salida.
 * @param num_samples_read: size_t* con el núemro de muestras leídas.
 */
TfLiteStatus generate_features(tflite::ErrorReporter* error_reporter,
                                   const int16_t* input, int input_size,
                                   int output_size, int8_t* output,
                                   size_t* num_samples_read);

#endif  // FEATURES_GENERATOR_H_
