#include "command_recognizer.h"

CommandRecognizer::CommandRecognizer(tflite::ErrorReporter* error_reporter,
                                     int32_t average_window_duration,
                                     uint8_t detection_threshold,
                                     int32_t suppression,
                                     int32_t minimum_count)
    : error_reporter_(error_reporter),
      average_window_duration_(average_window_duration),
      detection_threshold_(detection_threshold),
      suppression_(suppression),
      minimum_count_(minimum_count),
      previous_results_(error_reporter) {
  previous_top_label_ = "silence";
  previous_top_label_time_ = std::numeric_limits<int32_t>::min();
}

TfLiteStatus CommandRecognizer::process_latest_results(const TfLiteTensor* latest_results,
                                                       const int32_t current_time,
                                                       const char** found_command,
                                                       uint8_t* score, bool* is_new_command) {
  if ((latest_results->dims->size != 2) ||
      (latest_results->dims->data[0] != 1) ||
      (latest_results->dims->data[1] != nlabels)) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Las dimensiones de los resultados deben ser %d, pero son "
        "%d en con shape %d.", nlabels, latest_results->dims->data[1],
        latest_results->dims->size);
    return kTfLiteError;
  }

  if (latest_results->type != kTfLiteInt8) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Los resultados deberían ser int8_t, pero son %d",
        latest_results->type);
    return kTfLiteError;
  }

  if ((!previous_results_.empty()) &&
      (current_time < previous_results_.front().time_)) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "timestamp %d no es coherente con timestamp previa %d.",
        current_time, previous_results_.front().time_);
    return kTfLiteError;
  }

  previous_results_.push_back({current_time, latest_results->data.int8});

  const int64_t time_limit = current_time - average_window_duration_;
  while ((!previous_results_.empty()) &&
         previous_results_.front().time_ < time_limit) {
    previous_results_.pop_front();
  }

  const int64_t how_many_results = previous_results_.size();
  const int64_t earliest_time = previous_results_.front().time_;
  const int64_t samples_duration = current_time - earliest_time;
  if ((how_many_results < minimum_count_) ||
      (samples_duration < (average_window_duration_ / 4))) {
    *found_command = previous_top_label_;
    *score = 0;
    *is_new_command = false;
    return kTfLiteOk;
  }

  int32_t average_scores[nlabels];
  for (int offset = 0; offset < previous_results_.size(); ++offset) {
    PreviousResultsQueue::Result previous_result =
        previous_results_.from_front(offset);
    const int8_t* scores = previous_result.scores;
    for (int i = 0; i < nlabels; ++i) {
      if (offset == 0) {
        average_scores[i] = scores[i] + 128;
      } else {
        average_scores[i] += scores[i] + 128;
      }
    }
  }
  for (int i = 0; i < nlabels; ++i) {
    average_scores[i] /= how_many_results;
  }
  
  int current_top_index = 0;
  int32_t current_top_score = 0;
  for (int i = 0; i < nlabels; ++i) {
    if (average_scores[i] > current_top_score) {
      current_top_score = average_scores[i];
      current_top_index = i;
    }
  }
  const char* current_top_label = labels[current_top_index];

  int64_t time_since_last_top;
  if ((previous_top_label_ == labels[0]) ||
      (previous_top_label_time_ == std::numeric_limits<int32_t>::min())) {
    time_since_last_top = std::numeric_limits<int32_t>::max();
  } else {
    time_since_last_top = current_time - previous_top_label_time_;
  }
  if ((current_top_score > detection_threshold_) &&
      ((current_top_label != previous_top_label_) ||
       (time_since_last_top > suppression_))) {
    previous_top_label_ = current_top_label;
    previous_top_label_time_ = current_time;
    *is_new_command = true;
  } else {
    *is_new_command = false;
  }
  *found_command = current_top_label;
  *score = current_top_score;

  return kTfLiteOk;
}
