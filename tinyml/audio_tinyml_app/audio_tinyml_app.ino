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
// #include "premade_audio_model_data.h"
#include "feature_provider.h"
#include "command_recognizer.h"
#include "command_responder.h"

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
  int8_t* audio_input_buffer = nullptr;
  
  unsigned long last_inference_time;
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

// Inicialización del dispositivo.
void setup() {
  initialize_serial_port(9600, true);
  
  // Ajustamos error reporter.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Preparamos el modelo.
  audio_model = tflite::GetModel(audio_model_data);
  // audio_model = tflite::GetModel(premade_audio_model_data);
  if (audio_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "El modelo es de la versión %d, mientras que"
                         "la version soportada es %d.",
                         audio_model->version(), TFLITE_SCHEMA_VERSION);
    return; // Ha fallado el programa.
  }

  // Configuramos el OpsResolver.
  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // Preparamos el interpreter que ejecuta el modelo.
  static tflite::MicroInterpreter static_interpreter(audio_model, micro_op_resolver,
                                                     tensor_arena, kTensorArenaSize,
                                                     error_reporter);

  audio_interpreter = &static_interpreter;
  TfLiteStatus allocate_status = audio_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Fallo al asignar tensorArena");
    return;
  }

  // Preparamos punteros para ir a los buffers de input y output.
  audio_input = audio_interpreter->input(0);
  audio_output = audio_interpreter->output(0);
  audio_input_buffer = audio_input->data.int8;

  // Preparamos feature provider y el command_recognizer.
  static FeatureProvider static_feature_provider(elementCount, feature_buffer);
  audio_feature_provider = &static_feature_provider;

  static CommandRecognizer static_recognizer(error_reporter);
  audio_recognizer = &static_recognizer;

  previous_time = 0;
  last_inference_time = millis();
}

// Ciclo de ejecución del programa.
void loop() {
  const int32_t current_time = get_latest_audio_timestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = audio_feature_provider->populate_feature_data(error_reporter,
                                                                              previous_time,
                                                                              current_time,
                                                                              &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Fallo en la generación de features.");
    return;
  }

  previous_time = current_time;
  
  // Si no hay slices nuevas se omite iteración de loop.
  if (how_many_new_slices == 0) {
    return;
  }

  // Copiamos el feature buffer al input del modelo.
  for (int i = 0; i < elementCount; i++) {
    audio_input_buffer[i] = feature_buffer[i];
  }

  unsigned long t_ini = millis();
  TfLiteStatus invoke_status = audio_interpreter->Invoke();
  unsigned long t_end = millis();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Fallo en la inferencia.");
    return;
  }
  
  // Procesamos los resultados.
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = audio_recognizer->process_latest_results(audio_output,
                                                                         current_time,
                                                                         &found_command,
                                                                         &score,
                                                                         &is_new_command);
  if (process_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Fallo al procesar los resultados.");
    return;
  }
  
  respond_to_command(error_reporter, current_time, found_command, score, is_new_command);
  if (is_new_command) {
    TF_LITE_REPORT_ERROR(error_reporter, "Tiempo de inferencia: %dms\n"
                                         "Tiempo entre inferencias: %dms",
                         t_end - t_ini, t_end - last_inference_time);
  }
  last_inference_time = t_end;
}
