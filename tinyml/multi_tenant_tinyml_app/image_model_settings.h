/**
 * Parámetros utilizados en la creación del modelo
 * necesarios para interpretar los resultados.
==============================================================================*/

#ifndef MODEL_SETTINGS_H_
#define MODEL_SETTINGS_H_

constexpr int image_nlabels = 3;
constexpr int faceIndex = 0;
constexpr int maskIndex = 1;
constexpr int nothingIndex = 2;

extern const char* image_labels[image_nlabels];

#endif  // MODEL_SETTINGS_H_
