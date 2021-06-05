/*
 * Parámetros de configuración necesarios para ejecutar correctamente el
 * modelo.
==============================================================================*/

#ifndef AUDIO_MODEL_SETTINGS_H_
#define AUDIO_MODEL_SETTINGS_H_

// Tamaño del FFT que se usa para generar un espectrograma.
constexpr int maxAudioSampleSize = 512;
// Sample rate del microfono.
constexpr int audioSampleFrequency = 16000; // Hz

// Valores que indican las características del espectrograma generado.
constexpr int sliceSize = 40;
constexpr int sliceCount = 49;
constexpr int elementCount = (sliceSize * sliceCount);
constexpr int sliceStride = 20;
constexpr int sliceDuration = 30;

// Información del output del modelo
/*constexpr int audio_nlabels = 4;
constexpr int noIndex = 0;
constexpr int silenceIndex = 1;
constexpr int unknownIndex = 2;
constexpr int yesIndex = 3;*/

constexpr int audio_nlabels = 4;
constexpr int silenceIndex = 0;
constexpr int unknownIndex = 1;
constexpr int yesIndex = 2;
constexpr int noIndex = 3;

extern const char* audio_labels[audio_nlabels];

#endif  // AUDIO_MODEL_SETTINGS_H_
