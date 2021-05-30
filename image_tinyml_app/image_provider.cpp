#include "image_provider.h"

void initialize_camera() {
  // Inicializamos la cámara.
  if (!Camera.begin(QCIF, GRAYSCALE, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }
}

void prepare_image_data(int8_t* image_data) {
  // data es el buffer donde se guardan los datos de las imágenes capturadas.
  byte data[IMG_WIDTH * IMG_HEIGHT]; // QCIF: 176x144 en grayscale.
  Camera.readFrame(data);

  // Crop a 96x96 mientras se colocan los datois en input tensor.
  int x0 = (IMG_WIDTH - INPUT_WIDTH) / 2;
  int y0 = (IMG_HEIGHT - INPUT_HEIGHT) / 2;
  int i = 0;
  
  for (int y = y0; y < y0 + IMG_HEIGHT; y++) {
    for (int x = x0; x < x0 + IMG_WIDTH; x++) {
      image_data[i] = static_cast<int8_t>(data[(y * IMG_WIDTH) + x] * QUANT_FACTOR + QUANT_OFFSET);
      ++i;
    }
  }
}
