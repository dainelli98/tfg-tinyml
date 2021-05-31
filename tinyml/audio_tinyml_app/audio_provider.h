/*
 * Clase utilizada para extraer datos de audio del microfono integrado en el
 * microcontrolador.
==============================================================================*/

#ifndef AUDIO_PROVIDER_H_
#define AUDIO_PROVIDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "PDM.h"
#include "audio_model_settings.h"

#include "audio_provider.h"

/*
 * Registra nuevo audio para realizar inferencias.
 * @param error_reporter:     error reporter usado en la aplicación.
 * @param start:              int con el momento de inicio.
 * @param duration:           int con la duración de la muestra.
 * @param audio_samples_size: Puntero al tamaño de la muestra.
 * @param audio_samples:      Puntero a las muestras de audio.
 * @return  TFLiteStatus de la operación.
 */
TfLiteStatus get_audio_samples(tflite::ErrorReporter* error_reporter,
                               int start, int duration,
                               int* audio_samples_size,
                               int16_t** audio_samples);

/*
 * Devuelve el último tiempo en que se ha registrado audio.
 * @return  int32 con el último tiempo en que se ha registrado audio.
 */
int32_t get_latest_audio_timestamp();

#endif  // AUDIO_PROVIDER_H_
