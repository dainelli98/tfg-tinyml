/**
 * Esta clase se encarga de procesar los resultados de las inferencias
 * realizadas por el modelo para determinar si se ha pronunciado un comando.
==============================================================================*/
#ifndef COMMAND_RECOGNIZER_H_
#define COMMAND_RECOGNIZER_H_

#include <cstdint>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "audio_model_settings.h"

/*
 * Implementación básica de la clase queue para almacenar resultados
 * anteriores.
 */
class PreviousResultsQueue {
 public:
  PreviousResultsQueue(tflite::ErrorReporter* error_reporter)
      : error_reporter_(error_reporter), front_index_(0), size_(0) {}

  /*
   * Estructura de datos hecha para almacenar resultados de
   * inferencias.
   */
  struct Result {
    Result() : time_(0), scores() {}
    Result(int32_t tm, int8_t* input_scores) : time_(tm) {
      for (int i = 0; i < audio_nlabels; ++i) {
        scores[i] = input_scores[i];
      }
    }
    int32_t time_;
    int8_t scores[audio_nlabels];
  };

  /**
   * Devuelve el tamaño de la cola.
   * @return  int con el tamaño de la cola
   */
  int size() {
    return size_;
  }
  
  /*
   * Indica si la cola está vacía.
   * @return  true si la cola está vacía, false en caso contrario.
   */
  bool empty() {
    return size_ == 0;
  }

  /*
   * Devuelve el primer resultado de la cola.
   * @return  Result situado al inicio de la cola.
   */
  Result& front() {
    return results_[front_index_];
  }

  /**
   * Devuelve el último resultado de la cola.
   * @return  Result situado al final de la cola.
   */
  Result& back() {
    int back_index = front_index_ + (size_ - 1);
    if (back_index >= maxResults) {
      back_index -= maxResults;
    }
    return results_[back_index];
  }

  /**
   * Añade un resultado a la cola.
   * @param entry:  Result que se quiere añadir a la cola.
   */
  void push_back(const Result& entry) {
    if (size() >= maxResults) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "No caben más resultados en la cola.");
      return;
    }
    size_ += 1;
    back() = entry;
  }

  /*
   * Elimina el primer resultado de la cola y lo devuelve.
   * @return  Result situado al inicio de la cola.
   */
  Result pop_front() {
    if (size() <= 0) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "No hay resultados en la cola.");
      return Result();
    }
    Result result = front();
    front_index_ += 1;
    if (front_index_ >= maxResults) {
      front_index_ = 0;
    }
    size_ -= 1;
    return result;
  }

  /*
   * Helper para cceder otras posiciones de la cola.
   * @param offset: int con las posiciones respecto el inicio donde esta el
   *                resultado que se quiere obtener.
   * @param:  Result que se quiere obtener.
   */
  Result& from_front(int offset) {
    if ((offset < 0) || (offset >= size_)) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Attempt to read beyond the end of the queue!");
      offset = size_ - 1;
    }
    int index = front_index_ + offset;
    if (index >= maxResults) {
      index -= maxResults;
    }
    return results_[index];
  }

 private:
  tflite::ErrorReporter* error_reporter_;
  static constexpr int maxResults = 50;
  Result results_[maxResults];

  int front_index_;
  int size_;
};

/*
 * Esta clase permite interpretar una serie inferencias realizadas a lo largo del tiempo
 */
class CommandRecognizer {
 public:
  /*
   * Constructora de la clase CommandRecognizer.
   * @param error_reporter:           error_reporter usado en la aplicación.
   * @param average_window_duration:  int32_t con la duración media de una muestra en ms.
   * @param detection_threshold:      uint8_t con el threshold aplicado a la detección.
   * @param suppression:              int32_t con la supressión aplicada.
   * @paran minimum_count:            int32_t con el número de inferencias necesarias para
   *                                  confirmar la detección de un comando.
   */
  explicit CommandRecognizer(tflite::ErrorReporter* error_reporter,
                             int32_t average_window_duration = 1000,
                             uint8_t detection_threshold = 200,
                             int32_t suppression = 1500,
                             int32_t minimum_count = 3);

  /*
   * Actualiza los resultados de las predicciones resultadas.
   * @param latest_resutls: TfLiteTensor con los últimos resultados.
   * @param current_time:   int32_t con el timepo asociado a la última predicción.
   * @param found_command:  char** con el último comando detectado.
   * @param score:          uint8_t al que se le dará la score del último comando
   *                        detectado.
   * @param is_new_command: bool que indica si el último comando detectado es nuevo.
   * @return  TfLiteStatus indicando el resultado de la operación.
   */
  TfLiteStatus process_latest_results(const TfLiteTensor* latest_results,
                                      const int32_t current_time,
                                      const char** found_command, uint8_t* score,
                                      bool* is_new_command);

 private:
  // Configuración
  tflite::ErrorReporter* error_reporter_;
  int32_t average_window_duration_;
  uint8_t detection_threshold_;
  int32_t suppression_;
  int32_t minimum_count_;

  // Resultados anteriores obtenidos a lo largo del tiempo.
  PreviousResultsQueue previous_results_;
  // Última predicicón realizada.
  const char* previous_top_label_;
  // Tiempo de la última predicción realizada.
  int32_t previous_top_label_time_;
};

#endif  // COMMAND_RECOGNIZER_H_
