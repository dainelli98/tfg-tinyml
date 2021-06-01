#include "features_generator.h"

namespace {
  FrontendState g_micro_features_state;
  bool is_first_time = true;
}  // namespace

TfLiteStatus initialize_features(tflite::ErrorReporter* error_reporter) {
  FrontendConfig config;
  config.window.size_ms = sliceDuration;
  config.window.step_size_ms = sliceStride;
  config.noise_reduction.smoothing_bits = 10;
  config.filterbank.num_channels = sliceSize;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 7500.0;
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;
  if (!FrontendPopulateState(&config, &g_micro_features_state, audioSampleFrequency)) {
    TF_LITE_REPORT_ERROR(error_reporter, "FrontendPopulateState() ha fallado.");
    return kTfLiteError;
  }
  is_first_time = true;
  return kTfLiteOk;
}

void set_features_noise_estimates(const uint32_t* estimate_presets) {
  for (int i = 0; i < g_micro_features_state.filterbank.num_channels; ++i) {
    g_micro_features_state.noise_reduction.estimate[i] = estimate_presets[i];
  }
}

TfLiteStatus generate_features(tflite::ErrorReporter* error_reporter,
                                   const int16_t* input, int input_size,
                                   int output_size, int8_t* output,
                                   size_t* num_samples_read) {
  const int16_t* frontend_input;
  if (is_first_time) {
    frontend_input = input;
    is_first_time = false;
  }
  else {
    frontend_input = input + 160;
  }
  FrontendOutput frontend_output = FrontendProcessSamples(&g_micro_features_state,
                                                          frontend_input,
                                                          input_size,
                                                          num_samples_read);

  for (size_t i = 0; i < frontend_output.size; ++i) {
    constexpr int32_t value_scale = 256;
    constexpr int32_t value_div = static_cast<int32_t>((25.6f * 26.0f) + 0.5f);
    int32_t value = ((frontend_output.values[i] * value_scale) + (value_div / 2)) / value_div;
    //int32_t value = frontend_output.values[i] * QUANT_FACTOR + QUANT_OFFSET;
    value -= 128;
    if (value < -128) {
      value = -128;
    }
    if (value > 127) {
      value = 127;
    }
    output[i] = static_cast<int8_t>(value);
  }

  return kTfLiteOk;
}
