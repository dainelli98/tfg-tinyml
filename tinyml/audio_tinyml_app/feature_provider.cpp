#include "feature_provider.h"

FeatureProvider::FeatureProvider(int f_size, int8_t* f_data)
  : feature_size(f_size),
    feature_data(f_data),
    is_first_run(true) {
  for (int n = 0; n < feature_size; ++n) {
    feature_data[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}

TfLiteStatus FeatureProvider::populate_feature_data(tflite::ErrorReporter* error_reporter,
                                                    int32_t last_time,
                                                    int32_t current_time,
                                                    int* how_many_new_slices) {
  // Comprovamos que las medidas de los arrays.
  if (feature_size != elementCount) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Requested La longitud del feature vector %d no corresponde %d.",
                         feature_size, elementCount);
    return kTfLiteError;
  }

  // Solo actualizamos los nuevos fragmentos de audio.
  const int last_step = (last_time / sliceStride);
  const int current_step = (current_time / sliceStride);

  int slices_needed = current_step - last_step;
  
  // Si es la primera llamada a la función se limpia el feature array.
  if (is_first_run) {
    TfLiteStatus init_status = initialize_features(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    is_first_run = false;
    slices_needed = sliceCount;
  }
  if (slices_needed > sliceCount) {
    slices_needed = sliceCount;
  }
  *how_many_new_slices = slices_needed;

  const int slices_to_keep = sliceCount - slices_needed;
  const int slices_to_drop = sliceCount - slices_to_keep;

  if (slices_to_keep > 0) {
    for (int dest_slice = 0; dest_slice < slices_to_keep; ++dest_slice) {
      int8_t* dest_slice_data = feature_data + (dest_slice * sliceSize);
      const int src_slice = dest_slice + slices_to_drop;
      const int8_t* src_slice_data = feature_data + (src_slice * sliceSize);
      for (int i = 0; i < sliceSize; ++i) {
        dest_slice_data[i] = src_slice_data[i];
      }
    }
  }
  
  // Obtenemos los datos de audio correspondientes a las nuevas slices y
  // después se genera el espectrograma.
  if (slices_needed > 0) {
    for (int new_slice = slices_to_keep; new_slice < sliceCount; ++new_slice) {
      const int new_step = (current_step - sliceCount + 1) + new_slice;
      const int32_t slice_start = (new_step * sliceStride);
      int16_t* audio_samples = nullptr;
      int audio_samples_size = 0;
      get_audio_samples(error_reporter, (slice_start > 0 ? slice_start : 0),
                        sliceDuration, &audio_samples_size,
                        &audio_samples);
      if (audio_samples_size < maxAudioSampleSize) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Tamaño de datos de audio %d demasiado pequeño, deberia ser %d",
                             audio_samples_size, maxAudioSampleSize);
        return kTfLiteError;
      }
      int8_t* new_slice_data = feature_data + (new_slice * sliceSize);
      size_t num_samples_read;
      TfLiteStatus generate_status = generate_features(error_reporter, audio_samples,
                                                       audio_samples_size, sliceSize,
                                                       new_slice_data, &num_samples_read);
      if (generate_status != kTfLiteOk) {
        return generate_status;
      }
    }
  }
  return kTfLiteOk;
}
