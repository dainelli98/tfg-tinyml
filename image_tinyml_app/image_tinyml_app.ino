/*
 * Este programa para Arduino BLE 33 Sense utiliza el módulo de cámara
 * OV7675 para recoger muestras de imagen sobre los que se usa un modelo creado 
 * con TensorFlow para identifiacar la presencia de personas y si estas llevan
 * mascarilla.
 *
 * Desarrollado por Daniel Martín Martínez para la realización del
 * Trabajo de Fin de Grado titulado Aplicación de modelos de aprendizaje
 * automático en microcontroladores.
 */

#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "Arduino.h"

#include "image_model_data.h"
#include "image_provider.h"
#include "image_model_settings.h"
#include "image_responder.h"

#define DEBOUNCE_DELAY 50
#define BUTTON_PIN 13

// Las variable siguientes sirven para detectar cuando se pulsa el botón.
unsigned long lastDebounceTime;
bool lastButtonState;
bool buttonState;
bool do_inference;

// Variable globals, de uso habitual en aplicaciones TinyML.
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* image_model = nullptr;
  tflite::MicroInterpreter* image_interpreter = nullptr;
  TfLiteTensor* image_input = nullptr;
  TfLiteTensor* image_output = nullptr;
  
  // Arena para ejecutar el modelo.
  constexpr int kTensorArenaSize = 136 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
} // namespace

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

/**
 * Inicializa el botón del TinyML Shield.
 */
void initializeButton() {
  pinMode(BUTTON_PIN, OUTPUT);
  digitalWrite(BUTTON_PIN, HIGH);
  nrf_gpio_cfg(BUTTON_PIN,
                NRF_GPIO_PIN_DIR_OUTPUT,
                NRF_GPIO_PIN_INPUT_CONNECT,
                NRF_GPIO_PIN_PULLUP,
                NRF_GPIO_PIN_S0S1,
                NRF_GPIO_PIN_NOSENSE);
  lastDebounceTime = 0;
  lastButtonState = HIGH;
  do_inference = false;
}

/**
 * Detecta si se ha pulsado el botón de TinyML Shield.
 * @return                  true si se ha pulsado el botón, false en caso contrario.
 */
bool readShieldButton(){
  bool buttonRead = nrf_gpio_pin_read(digitalPinToPinName(BUTTON_PIN));
   
  if (buttonRead != lastButtonState) {
    lastDebounceTime = millis();
  }
 
  if (millis() - lastDebounceTime >= DEBOUNCE_DELAY) {
    if (buttonRead != buttonState) {
      buttonState = buttonRead;
 
      if (!buttonState) {
        lastButtonState = buttonRead;
        return true;
      }
    }
  }
 
  lastButtonState = buttonRead;
  return false;
}

// Device initialization.
void setup() {
  initialize_serial_port(9600, true);
  
  initialize_camera();

  initializeButton();

  initialize_responder();

  // Ajustamos error reporter.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Preparamos el modelo.
  image_model = tflite::GetModel(image_model_data);
  if (image_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "El modelo es de la versión %d, mientras que"
                         "la version soportada es %d.",
                         image_model->version(), TFLITE_SCHEMA_VERSION);
    return; // Ha fallado el programa.
  }

  // Configuramos el OpsResolver.
  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddReshape();
  // micro_op_resolver.AddRelu();
  micro_op_resolver.AddRelu6();

  // Preparamos el interpreter que ejecuta el modelo
  static tflite::MicroInterpreter static_interpreter(image_model, micro_op_resolver,
                                                     tensor_arena, kTensorArenaSize,
                                                     error_reporter);
  image_interpreter = &static_interpreter;
  TfLiteStatus allocate_status = image_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Fallo al asignar tensorArena");
    return;
  }

  // Preparamos pointer al input tensor.
  image_input = image_interpreter->input(0);
  image_output = image_interpreter->output(0);
}

// Ciclo de ejecución del programa.
void loop() {
  if (readShieldButton()) {
    do_inference = not do_inference;
    if (do_inference)
      Serial.println("Inferencia iniciada.");
    else
      Serial.println("Inferencia detenida.");  
  }

  if (do_inference) {
    
    // Se preparan los datos de imagen en el input tensor del interpreter.
    prepare_image_data(image_input->data.int8);
  
    // Ejecutamos la inferencia sobre los datos de imagen.
    if (kTfLiteOk != image_interpreter->Invoke()) {
      TF_LITE_REPORT_ERROR(error_reporter, "Error al realizar inferencia.");
    }
  
    TfLiteTensor* output = image_output;
  
    // Extraemos los resultados.
    int8_t face_score = output->data.int8[faceIndex];
    int8_t mask_score = output->data.int8[maskIndex];
    int8_t nothing_score = output->data.int8[nothingIndex];
  
    // Se reliza una respuesta a la inferencia realizada.
    respond_image_inference(error_reporter, face_score, mask_score, nothing_score);
  }
}
