/*
 Este programa sirve para realizar el muestreo de imagenes requerido para el
 desarrollo de aplicaciones TinyML, a través del microcontrolador Arduino
 BLE 33 Sense y su módulom de camara OV7675.
 
 Desarrollado por Daniel Martín Martínez para la realización del
 Trabajo de Fin de Grado titulado Aplicación de modelos de aprendizaje
 automático en microcontroladores.
 Este programa se basa en el condigo de CameraVisualizerRawBytes que se presenta
 en el artículo:
 https://blog.arduino.cc/2020/06/24/machine-vision-with-low-cost-camera-modules/
*/

import processing.serial.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.io.File;
import java.util.Arrays;
import java.util.Scanner;

Serial myPort;
int selectedClass;

// Ajustamos la resolución en función de los ajustes de captura del microcontrolador.
final int cameraWidth = 176;
final int cameraHeight = 144;
final int cameraBytesPerPixel = 2;
final int bytesPerFrame = cameraWidth * cameraHeight * cameraBytesPerPixel;

// Carpeta donde se guardan las imagenes
final String path = "/home/daniel/Documentos/TFG/Samples";

String sessionName;

PImage myImage;

// Creamos un buffer para guardar los datos de imagen recibidos.
byte[] frameBuffer = new byte[bytesPerFrame];

// Indicamos el número de clases y sus nombres.
final int nclasses = 3;
final String[] classes = {"face", "mask", "nothing"};

// Añadimos contadores para dar nombre a las imagenes;
int[] counters = new int[nclasses];

/**
 Pregunta al usuario a que clase pertenece la imagen tomada.
 @returns  int que indica la classe de la imagen tomada.
*/
void ask_class() {
  print("Indica la clase de la imagen que tomarás a continuación pulsando el botón indicado:\n\t0 - No guardar");
  for (int i = 0; i < nclasses; ++i) {
    print("\n\t" + String.valueOf(i + 1) + " - " + classes[i]);
  }
  print("\n");
}


/**
 Permite indidcar la classe sobre la que se recogen las muestras mediante las teclas del teclado.
*/
void keyPressed(){
  if (key == '1' && nclasses >= 1) {
    selectedClass = 0;
  }
  else if (key == '2' && nclasses >= 2) {
    selectedClass = 1;
  }
  else if (key == '3' && nclasses >= 3) {
    selectedClass = 2;
  }
  else if (key == '4' && nclasses >= 4) {
    selectedClass = 3;
  }
  else if (key == '5' && nclasses >= 5) {
    selectedClass = 4;
  }
  else if (key == '6' && nclasses >= 6) {
    selectedClass = 5;
  }
  else if (key == '7' && nclasses >= 7) {
    selectedClass = 6;
  }
  else if (key == '8' && nclasses >= 8) {
    selectedClass = 7;
  }
  else if (key == '9' && nclasses >= 9) {
    selectedClass = 8;
  }
  else
    selectedClass = -1;
  if (selectedClass == -1)
    println("Las siguientes capturas no se guardarán");
  else
    println("Las siguientes capturas se asignarán a la clase: " + classes[selectedClass]);
}

void setup()
{
  // Damos nombre a la sessión
  sessionName = String.valueOf(day()) + "-" + String.valueOf(month()) + "-" + String.valueOf(year());
  
  // Ajustamos el tamaño del visor (debería ser de dimensiones iguales a las imagenes que se esperan).
  size(1584, 1296);
  
  // Inicializamos contadores.
  Arrays.fill(counters, 1);
  
  // Indicamos el puerto donde se esperarán los datos de imagen.
  myPort = new Serial(this, Serial.list()[0], 9600);           // Si sólo hay un puerto Serial activo.
  //myPort = new Serial(this, "COM5", 9600);                   // Windows
  //myPort = new Serial(this, "/dev/ttyACM0", 9600);           // Linux
  //myPort = new Serial(this, "/dev/cu.usbmodem14401", 9600);  // Mac

  // wait for full frame of bytes
  myPort.buffer(bytesPerFrame);  
  
  selectedClass = -1;

  myImage = createImage(cameraWidth, cameraHeight, RGB);
  
  ask_class();
}

void draw()
{
  image(myImage, 0, 0, 1584, 1296);
}

void serialEvent(Serial myPort) {
  // Leemos los datos que llegan a través del puerto Serial.
  myPort.readBytes(frameBuffer);
  
  /**for (int i= 0; i < bytesPerFrame; ++i) {
    print(frameBuffer[i]);
    print(',');
  }*/
  
  // Accedemos a los bytes mediante un ByteBuffer.
  ByteBuffer bb = ByteBuffer.wrap(frameBuffer);
  bb.order(ByteOrder.BIG_ENDIAN);

  int i = 0;

  while (bb.hasRemaining()) {
    // read 16-bit pixel
    short p = bb.getShort();

    // Conversión de RGB565 a RGB 24-bit
    int r = ((p >> 11) & 0x1f) << 3;
    int g = ((p >> 5) & 0x3f) << 2;
    int b = ((p >> 0) & 0x1f) << 3;

    // set pixel color
    myImage.pixels[i++] = color(r, g, b);
  }
  myImage.updatePixels();
  
  if (selectedClass >= 0 && selectedClass < nclasses) {
    String filepath = path + '/' + sessionName + "--" + String.valueOf(classes[selectedClass]) + String.valueOf(counters[selectedClass]) + ".jpg";
    while (new File(filepath).exists()) {
      ++counters[selectedClass];
      filepath = path + '/' + sessionName + "--" + String.valueOf(classes[selectedClass]) + String.valueOf(counters[selectedClass]) + ".jpg";
    }
    myImage.save(filepath);
    ++counters[selectedClass];
  }
}
