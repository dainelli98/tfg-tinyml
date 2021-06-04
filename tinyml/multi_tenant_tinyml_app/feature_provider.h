/*
 * Clase utilizada para extraer las características, ya preprocesadas del audio
 * sobre el que se realizan las inferencias.
==============================================================================*/

#ifndef FEATURE_PROVIDER_H_
#define FEATURE_PROVIDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"


#include "audio_provider.h"
#include "features_generator.h"
#include "audio_model_settings.h"

/*
 * Clase utilizada para extraer las características, ya preprocesadas del audio
 * sobre el que se realizan las inferencias.
 */
class FeatureProvider {
 public:

  /*
   * Constructora de la clase FeatureProvider.
   * @param f_size: int con la longitud del feature data array.
   * @param f_data: Puntero al array que contiene los datos que se dan
   *                como input al modelo.
   */
  FeatureProvider(int f_size, int8_t* f_data);
  ~FeatureProvider();

  /*
   * Actualiza los datos del feature data array que se dan como entrada del
   * modelo.
   * @param error_reporter:       error reporter usado en la aplicación.
   * @param last_time:            int32_t con el momento en el que se ejecutó
   *                              esta función la última vez.
   * @param current_time:         int32_t con el momento en el que se está
   *                              ejecutando esta función.
   * @param how_many_new_slices:  Puntero a la variable que contendrá el
   *                              número de ventanas actualizadas con la
   *                              ejecución de la función.
   * @return  TFLiteStatus de la operación.
   */
  TfLiteStatus populate_feature_data(tflite::ErrorReporter* error_reporter,
                                     int32_t last_time, int32_t current_time,
                                     int* how_many_new_slices);

 private:
  // int con la longitud del feature data array.
  int feature_size;
  // Puntero al array que contiene los datos que se dan como input al modelo.
  int8_t* feature_data;
  // bool que indica si aun no se han cargado ningunos datos aun.
  bool is_first_run;
};

#endif  // FEATURE_PROVIDER_H_
