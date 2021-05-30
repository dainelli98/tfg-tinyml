#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "image_responder.h"

void initialize_responder() {
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDB, HIGH);
}

void respond_image_inference(tflite::ErrorReporter* error_reporter,
                             int8_t face_score, int8_t mask_score,
                             int8_t nothing_score) {

  if (face_score > mask_score and face_score > nothing_score) { // Detectada clase face.
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, LOW);
  }
  else if (mask_score > nothing_score) {  // Detectada clase mask.
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDB, HIGH);
  }
  else  // Detectada clase nothing.
  {
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
  }

  TF_LITE_REPORT_ERROR(error_reporter, "---------------------\nface score:\t%d\nmask score:\t%d\nnothing_score:\t%d",
                       face_score, mask_score, nothing_score);
  delay(500);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDB, HIGH);
  
}

#endif  // ARDUINO_EXCLUDE_CODE
