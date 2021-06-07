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

// Defines para el botón.
#define DEBOUNCE_DELAY  50
#define BUTTON_PIN      13

// Defines de los estados en que puede estar la aplicación.
#define LISTEN_COMMAND  0
#define SCAN_FACE_ENTER 1
#define SCAN_FACE_EXIT  2

// Definicion aforo máximo.
#define MAX_AFORO       10

// Definicion timeout para encontrar cara.
#define FACE_TIMEOUT    10000  // ms

// Definición de si se necesitat mascara para salir.
#define MASK_TO_EXIT    false

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
  int32_t previous_time = 0;
  
  // Arena para ejecutar el modelo.
  constexpr int kTensorArenaSize = 42 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];

  // Las variable siguientes sirven para detectar cuando se pulsa el botón.
  unsigned long lastDebounceTime;
  bool lastButtonState;
  bool buttonState;
  bool do_inference;

  // Variable para registrar numero de presonas en el recinto.
  short int ocupacion;

  // Variable para almacenar el estado actual de la aplicación.
  short int state;

  // Variable para poder usar timeout.
  unsigned long timeout_start;
  unsigned long last_inference_time;
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

  // Ajustamos error reporter.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Creamos allocator que será compartido por los 2 modelos.
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(tensor_arena,
                                                                     kTensorArenaSize,
                                                                     error_reporter);

  // Configuramos el OpsResolver para los 2 modelos.
  static tflite::MicroMutableOpResolver<8> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddRelu6();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddSoftmax();
  
  // Preparamos el modelo de imagen.
  image_model = tflite::GetModel(image_model_data);
  if (image_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "El modelo de imagen es de la versión %d, mientras que"
                         "la version soportada es %d.",
                         image_model->version(), TFLITE_SCHEMA_VERSION);
    return; // Ha fallado el programa.
  }

  // Preparamos el interpreter que ejecuta el modelo
  static tflite::MicroInterpreter image_static_interpreter(image_model, micro_op_resolver,
                                                           allocator, error_reporter);
  image_interpreter = &image_static_interpreter;
  
  TfLiteStatus image_allocate_status = image_interpreter->AllocateTensors();
  if (image_allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Fallo al asignar tensorArena");
    return;
  }

  // Preparamos pointer al input y output tensor del modelo de imagen.
  image_input = image_interpreter->input(0);
  image_output = image_interpreter->output(0);

  // Cargamos el modelo de audio.
  audio_model = tflite::GetModel(premade_audio_model_data);
  if (audio_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "El modelo de audio es de la versión %d, mientras que"
                         "la version soportada es %d.",
                         audio_model->version(), TFLITE_SCHEMA_VERSION);
    return; // Ha fallado el programa.
  }

  // Preparamos el interpreter que ejecuta el modelo de audio
  static tflite::MicroInterpreter audio_static_interpreter(audio_model, micro_op_resolver,
                                                     allocator, error_reporter);

  audio_interpreter = &audio_static_interpreter;

  TfLiteStatus audio_allocate_status = audio_interpreter->AllocateTensors();
  if (audio_allocate_status != kTfLiteOk) {
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
  state = LISTEN_COMMAND;
  timeout_start = millis();
  last_inference_time = millis();
}

// Ciclo de ejecución del programa.
void loop() {
  if (state == LISTEN_COMMAND) {
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

    // Realizamos inferencia usando el modelo de audio.
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
    
    // Mostramos resultados de inferencia.
    respond_to_command(error_reporter, current_time, found_command, score, is_new_command);
    if (is_new_command) {
      TF_LITE_REPORT_ERROR(error_reporter, "Tiempo de inferencia: %dms\n"
                                           "Tiempo entre inferencias: %dms",
                           t_end - t_ini, t_end - last_inference_time);
    }
    last_inference_time = t_end;
    
    // Modificamos el estado de la aplicación en función de los resultados obtenidos en la inferencia.
    bool button = readShieldButton();
    if ((is_new_command && found_command[0] != 'u' && found_command[0] != 's') || button) {
      if (found_command[0] == 'y' || button) {
        if (ocupacion >= MAX_AFORO) {
          TF_LITE_REPORT_ERROR(error_reporter, "Se ha alcanzado ya el aforo máximo: %d.",
                               MAX_AFORO);
          if (button) {
            TF_LITE_REPORT_ERROR(error_reporter, "Se ha forzado entrada con botón. Se reduce ocupación en 1.");
            --ocupacion;
          }
        }
        TF_LITE_REPORT_ERROR(error_reporter, "Detectado comando de entrada.\nOcupación actual: %d\nAforo máximo: %d"
                                             "\nInciando escaneo facial de entrada.", ocupacion, MAX_AFORO);
        state = SCAN_FACE_ENTER;
        timeout_start = millis();
      }
      else if (found_command[0] == 'n') {
        TF_LITE_REPORT_ERROR(error_reporter, "Detectado comando de salida.\nOcupación actual: %d\nAforo máximo: %d"
                                             "\nInciando escaneo facial de salida.", ocupacion, MAX_AFORO);
        state = SCAN_FACE_EXIT;
        timeout_start = millis();
      }
    }
  }
  else {
    // Se preparan los datos de imagen en el input tensor del interpreter.
    prepare_image_data(image_input->data.int8);
  
    // Ejecutamos la inferencia sobre los datos de imagen.
    unsigned long t_ini = millis();
    TfLiteStatus inference_status = image_interpreter->Invoke();
    unsigned long t_end = millis();
    if (kTfLiteOk != inference_status) {
      TF_LITE_REPORT_ERROR(error_reporter, "Error al realizar inferencia.");
    }
  
    TfLiteTensor* output = image_output;
  
    // Extraemos los resultados.
    int8_t face_score = output->data.int8[faceIndex];
    int8_t mask_score = output->data.int8[maskIndex];
    int8_t nothing_score = output->data.int8[nothingIndex];
  
    // Se realiza una respuesta a la inferencia realizada.
    respond_image_inference(error_reporter, face_score, mask_score, nothing_score);
    TF_LITE_REPORT_ERROR(error_reporter, "Tiempo de inferencia: %dms\n"
                                         "Tiempo entre inferencias: %dms",
                         t_end - t_ini, t_end - last_inference_time);
    last_inference_time = t_end;

    if (state == SCAN_FACE_ENTER) {
      if (mask_score > nothing_score && mask_score > face_score) {
        ++ocupacion;
        TF_LITE_REPORT_ERROR(error_reporter, "Se ha identificado una cara con mascarilla."
                                             "\nSe permite la entrada.\nOcupación: %d/%d", ocupacion, MAX_AFORO);
        state = LISTEN_COMMAND;
      }
    }
    
    else {
      if ((mask_score > nothing_score && mask_score > face_score) || (face_score > nothing_score && !MASK_TO_EXIT)) {
        --ocupacion;
        TF_LITE_REPORT_ERROR(error_reporter, "Se ha identificado una cara."
                                             "\nSe permite la salida.\nOcupación: %d/%d", ocupacion, MAX_AFORO);
        state = LISTEN_COMMAND;
      }
    }

    if (millis() >= timeout_start + FACE_TIMEOUT && state != LISTEN_COMMAND) {
      state = LISTEN_COMMAND;
      TF_LITE_REPORT_ERROR(error_reporter, "No se ha detectado una cara dentro del limite de tiempo establecido."
                                           "\nSe vuelve a la espera de comando de entrada o salida.");
    }
  }
}
