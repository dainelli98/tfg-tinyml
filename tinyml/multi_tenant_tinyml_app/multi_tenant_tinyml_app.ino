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

#include "audio_model_settings.h"
// #include "audio_model_data.h"
#include "premade_audio_model_data.h"
#include "feature_provider.h"
#include "command_recognizer.h"
#include "command_responder.h"

#define DEBOUNCE_DELAY 50
#define BUTTON_PIN 13


// Variables globales.
namespace {
  // Variables para cargar y ejecutar el modelo.
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* image_model = nullptr;
  const tflite::Model* audio_model = nullptr;
  tflite::MicroInterpreter* image_interpreter = nullptr;
  tflite::MicroInterpreter* audio_interpreter = nullptr;
  TfLiteTensor* image_input = nullptr;
  TfLiteTensor* image_output = nullptr;
  TfLiteTensor* audio_input = nullptr;
  TfLiteTensor* audio_output = nullptr;

  FeatureProvider* audio_feature_provider = nullptr;
  CommandRecognizer* audio_recognizer = nullptr;
  int8_t feature_buffer[elementCount];
  int8_t* audio_input_buffer = nullptr;
  
  // Arena para ejecutar el modelo.
  constexpr int kTensorArenaSize = 41 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];

  // Las variable siguientes sirven para detectar cuando se pulsa el botón.
  unsigned long lastDebounceTime;
  bool lastButtonState;
  bool buttonState;
  bool do_inference;
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

// Inicialización del dispositivo.
void setup() {
  initialize_serial_port(9600, true);
  
  initialize_camera();

  initializeButton();

  initialize_responder();

}

// Ciclo de ejecución del programa.
void loop() {

}
