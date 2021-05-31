/*
 * Este programa para Arduino BLE 33 Sense utiliza el módulo de cámara
 * OV7675 para recoger muestras de imagen que se envían a un script de
 * processing que los guarda para ser utilizados en el desarrollo de
 * modelos de TinyML. 
 *
 * Desarrollado por Daniel Martín Martínez para la realización del
 * Trabajo de Fin de Grado titulado Aplicación de modelos de aprendizaje
 * automático en microcontroladores.
 * Este programa se basa en el ejemplo de la librería Arduino_OV767X que
 * se presenta en el artículo:
 * https://blog.arduino.cc/2020/06/24/machine-vision-with-low-cost-camera-modules/
 * El código para detectar cuando se usa el botón de la placa procede de
 * la librería del curso de edX sobre TinyML:
 * https://github.com/tinyMLx/arduino-library/blob/main/src/TinyMLShield.h
===================================================================================*/

#include <Arduino_OV767X.h>

#define BUTTON_PIN 13

// data es el buffer donde se guardan los datos de las imágenes capturadas.
byte data[176 * 144 * 2]; // QCIF: 176x144 X 2 bytes por píxel (RGB565)

// Las variable siguientes sirven para detectar cuando se pulsa el botón.
unsigned long lastDebounceTime;
unsigned long debounceDelay;
bool lastButtonState;
bool buttonState;


// Inicializamos la variable que guarda el numero de bytes por captura.
int bytesPerFrame;

/**
 * Inicializa el botón del TinyML Shield.
 */
void initializeButton() {
  pinMode(BUTTON_PIN, OUTPUT);
  digitalWrite(BUTTON_PIN, HIGH);
  nrf_gpio_cfg(BUTTON_PIN,
                NRF_GPIO_PIN_DIR_OUTPUT,
                NRF_GPIO_PIN_INPUT_CONNECT,
                NRF_GPIO_PIN_PULLUP,
                NRF_GPIO_PIN_S0S1,
                NRF_GPIO_PIN_NOSENSE);
  lastDebounceTime = 0;
  debounceDelay = 50;
  lastButtonState = HIGH;
}

/**
 * Detecta si se ha pulsado el botón de TinyML Shield
 * @returns true si se ha pulsado el botón, false en caso contrario.
 */
bool readShieldButton(){
  bool buttonRead = nrf_gpio_pin_read(digitalPinToPinName(BUTTON_PIN));
   
  if (buttonRead != lastButtonState) {
    lastDebounceTime = millis();
  }
 
  if (millis() - lastDebounceTime >= debounceDelay) {
    if (buttonRead != buttonState) {
      buttonState = buttonRead;
 
      if (!buttonState) {
        lastButtonState = buttonRead;
        return true;
      }
    }
  }
 
  lastButtonState = buttonRead;
  return false;
}


// Device initialization.
void setup() {
  // Iniciamos comunicación con puerto Serial.
  Serial.begin(9600);

  // Esperamos a que se active el Serial Monitor.
  while (!Serial);

  // Inicializamos el botón de TinyML Shield.
  initializeButton();

  // Inicializamos la cámara.
  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }

  // Guardamos el número de bytes que contiene un frame.
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
}

void loop() {
  // Esperamos a que se pulse el botón.
  if (readShieldButton()) {
    // Capturamos una imagen.
    Camera.readFrame(data);

    // Enviamos la imagen a través del serial monitor.
    Serial.write(data, bytesPerFrame);
  }
}
