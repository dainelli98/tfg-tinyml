/**
 * Este es el modelo de audio TensorFlow Lite convertido en array de C,
 * para poder ser usado en el microcontrolador sacado de los ejemplos de
 * TensorFlow.
==============================================================================*/

#ifndef PREMADE_AUDIO_MODEL_DATA_H_
#define PREMADE_AUDIO_MODEL_DATA_H_

// Array de bytes que contiene el modelo de audio entrenado.
extern const unsigned char premade_audio_model_data[];

// unsigned int que indica la longitud del array audio_model_data.
extern const unsigned int premade_audio_model_data_len;

#endif  // PREMADE_AUDIO_MODEL_DATA_H_
