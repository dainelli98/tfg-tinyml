/*
 * Este programa para Arduino BLE 33 Sense utiliza el microfono integrado
 * para recoger una serie continua de audio sobre la que se usa un modelo
 * creado con TensorFlow para identificar comandos de voz.
 * 
 * Se utiliza el código ejemplo proporcionado por el equipo de TensorFlow
 * Para generar los espectrogramas:
 * https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/micro_speech
 *
 * Desarrollado por Daniel Martín Martínez para la realización del
 * Trabajo de Fin de Grado titulado Aplicación de modelos de aprendizaje
 * automático en microcontroladores.
==============================================================================*/

#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "Arduino.h"

#include "audio_model_settings.h"
#include "audio_model_data.h"
#include "feature_provider.h"
#include "command_recognizer.h"

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* audio_model = nullptr;
  tflite::MicroInterpreter* audio_interpreter = nullptr;
  TfLiteTensor* audio_input = nullptr;
  TfLiteTensor* audio_output = nullptr;
  FeatureProvider* audio_feature_provider = nullptr;
  CommandRecognizer* audio_recognizer = nullptr;
  int32_t previous_time = 0;
  
  constexpr int kTensorArenaSize = 10 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  int8_t feature_buffer[elementCount];
  int8_t* model_input_buffer = nullptr;
}  // namespace

/**
 * Inicializa la conexión con serial port.
 * @param baud_rate:  int con el baud rate que se usará en la connexión
 *                    con serial port.
 * @param wait:       bool que si es True hace que el dispositivo espere
 *                    a detectar la conexión con serial monitor.
 */
void initialize_serial_port(int baud_rate, bool wait) {
  // Iniciamos comunicación con puerto Serial.
  Serial.begin(baud_rate);

  if (wait) {
    // Esperamos a que se active el Serial Monitor.
    while (!Serial);
  }

  Serial.println("Serial port inicializado.");
}

void setup() {
  initialize_serial_port(9600, true);
  
  // Ajustamos error reporter.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Preparamos el modelo.
  audio_model = tflite::GetModel(audio_model_data);
  if (audio_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "El modelo es de la versión %d, mientras que"
                         "la version soportada es %d.",
                         audio_model->version(), TFLITE_SCHEMA_VERSION);
    return; // Ha fallado el programa.
  }
}

void loop() {
}
