/**
 * Este es el modelo de imagen TensorFlow Lite convertido en array de C,
 * para poder ser usado en el microcontrolador.
==============================================================================*/

#ifndef IMAGE_MODEL_DATA_H_
#define IMAGE_MODEL_DATA_H_

// Array de bytes que contiene el modelo de imagen entrenado.
extern const unsigned char image_model_data[];

// unsigned int que indica la longitud del array image_model_data.
extern const unsigned int image_model_data_len;

#endif  // IMAGE_MODEL_DATA_H_
