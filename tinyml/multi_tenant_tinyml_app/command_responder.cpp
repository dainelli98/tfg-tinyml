#include "command_responder.h"

void respond_to_command(tflite::ErrorReporter* error_reporter,
                        int32_t current_time, const char* found_command,
                        uint8_t score, bool is_new_command) {
  static bool is_initialized = false;
  
  // Si no se ha hecho aun se inicializan los LED del microcontrolador.
  if (!is_initialized) {
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
    is_initialized = true;
  }
  static int32_t last_command_time = 0;
  static int count = 0;
  static int certainty = 220;

  if (is_new_command) {
    
    digitalWrite(LED_BUILTIN, LOW);
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
    
    // Encendemos led correspondiente al comando.
    if (found_command[0] == 'y') {  // yes
      last_command_time = current_time;
      digitalWrite(LEDG, LOW);
    }

    else if (found_command[0] == 'n') {  // no
      last_command_time = current_time;
      digitalWrite(LEDR, LOW);
    }

    else if (found_command[0] == 'u') {  // unknown
      last_command_time = current_time;
      digitalWrite(LEDB, LOW);
    }

    if (found_command[0] != 's')
      TF_LITE_REPORT_ERROR(error_reporter, "---------------------\nDetectado: %s\nPuntuación: %d\nTiempo: @%dms",
                           found_command, score, current_time);
  }

  // Apagamos LED al rato.
  if (last_command_time != 0) {
    if (last_command_time < (current_time - 500)) {
      last_command_time = 0;
      digitalWrite(LED_BUILTIN, LOW);
      digitalWrite(LEDR, HIGH);
      digitalWrite(LEDG, HIGH);
      digitalWrite(LEDB, HIGH);
    }
    return;
  }

  ++count;
  if (count & 1) {
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    digitalWrite(LED_BUILTIN, LOW);
  }
}
