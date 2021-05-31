#include "audio_provider.h"

namespace {
  bool is_audio_initialized = false;
  constexpr int audioCaptureBufferSize = DEFAULT_PDM_BUFFER_SIZE * 16;
  int16_t audio_capture_buffer[audioCaptureBufferSize];
  int16_t audio_output_buffer[maxAudioSampleSize];
  volatile int32_t latest_audio_timestamp = 0;
} // namespace

void capture_samples() {
  const int number_of_samples = DEFAULT_PDM_BUFFER_SIZE / 2;
  const int32_t time_in_ms = latest_audio_timestamp + (number_of_samples /
                                                       (audioSampleFrequency
                                                        / 1000));

  const int32_t start_sample_offset = latest_audio_timestamp * (audioSampleFrequency /
                                                                1000);
  
  const int capture_index = start_sample_offset % audioCaptureBufferSize;
  
  PDM.read(audio_capture_buffer + capture_index, DEFAULT_PDM_BUFFER_SIZE);
  
  latest_audio_timestamp = time_in_ms;
}

TfLiteStatus init_audio_recording(tflite::ErrorReporter* error_reporter) {
  PDM.onReceive(capture_samples);
  PDM.begin(1, audioSampleFrequency);
  PDM.setGain(20);
  
  while (!latest_audio_timestamp) {
  }

  return kTfLiteOk;
}

TfLiteStatus get_audio_samples(tflite::ErrorReporter* error_reporter,
                               int start, int duration,
                               int* audio_samples_size, int16_t** audio_samples) {
  // Inicializamos si no se ha hecho préviamente.
  if (!is_audio_initialized) {
    TfLiteStatus init_status = init_audio_recording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    is_audio_initialized = true;
  }

  const int start_offset = start * (audioSampleFrequency / 1000);

  const int duration_sample_count = duration * (audioSampleFrequency / 1000);
  for (int i = 0; i < duration_sample_count; ++i) {
    const int capture_index = (start_offset + i) % audioCaptureBufferSize;
    audio_output_buffer[i] = audio_capture_buffer[capture_index];
  }

  *audio_samples_size = maxAudioSampleSize;
  *audio_samples = audio_output_buffer;

  return kTfLiteOk;
}

int32_t get_latest_audio_timestamp() {
  return latest_audio_timestamp;
}
