/**
 * Función encargada actuar de forma correcta en función de las inferencias
 * realizadas.
==============================================================================*/


#ifndef COMMAND_RESPONDER_H_
#define COMMAND_RESPONDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "Arduino.h"

/*
 * Ejecuta la respuesta adecuada a una inferencia.
 * @param error_reporter: error reporter usado en la aplicación.
 * @param current_time:   int32_t con el momento actual. 
 * @param found_command:  const char* con el comando reconocido.
 * @param score:          uint8_t con la puntuación del comando reconocido.
 * @param is_new_command: bool que indica si el comando reconocido es nuevo.
 */
void respond_to_command(tflite::ErrorReporter* error_reporter,
                        int32_t current_time, const char* found_command,
                        uint8_t score, bool is_new_command);

#endif  // COMMAND_RESPONDER_H_
 
