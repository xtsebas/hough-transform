/*
 ============================================================================
 Author        : G. Barlas (Modificado por estudiante)
 Version       : 1.1
 Last modified : Noviembre 2025
 License       : Released under the GNU GPL 3.0
 Description   : Implementación de Transformada de Hough Lineal en CUDA
                 Versión con Memoria Global (baseline)
 ============================================================================
 */
#include "../include/image_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#include <vector>

// ============================================================================
// CONSTANTES GLOBALES
// ============================================================================
const int degreeInc = 2;                    // Incremento en grados para el ángulo theta
const int degreeBins = 180 / degreeInc;     // Número de bins para theta (90 bins)
const int rBins = 100;                      // Número de bins para la distancia r
const float radInc = degreeInc * M_PI / 180;// Incremento en radianes

//*****************************************************************
// IMPLEMENTACIÓN CPU (REFERENCIA)
// Función secuencial que sirve como baseline para verificar resultados
//*****************************************************************
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  // Calcular radio máximo: distancia del centro a la esquina
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  
  // Crear acumulador: matriz de votación [rBins x degreeBins]
  *acc = new int[rBins * degreeBins];
  memset (*acc, 0, sizeof (int) * rBins * degreeBins);
  
  // Calcular centro de la imagen (origen del sistema de coordenadas)
  int xCent = w / 2;
  int yCent = h / 2;
  
  // Factor de escala para mapear r al rango de bins
  float rScale = 2 * rMax / rBins;

  // Iterar sobre cada píxel de la imagen
  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
      {
        int idx = j * w + i;
        if (pic[idx] > 0) // Si el píxel está "encendido" (blanco)
          {
            // Convertir a coordenadas centradas
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // Y invertida para sistema cartesiano
            float theta = 0;         // Ángulo actual en radianes
            
            // Para cada posible ángulo, calcular r y votar
            for (int tIdx = 0; tIdx < degreeBins; tIdx++)
              {
                // Ecuación de la línea: r = x*cos(θ) + y*sin(θ)
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; // Incrementar voto
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// KERNEL GPU - VERSIÓN CON MEMORIA GLOBAL
// Un thread por píxel de la imagen
// El acumulador está en memoria global
//*****************************************************************
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, 
                                float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  /*============================================================================
   * CÁLCULO DEL ID GLOBAL DEL THREAD
   * 
   * Configuración del kernel: <<< blockNum, 256 >>>
   * - Grid 1D: blockNum bloques
   * - Block 1D: 256 threads por bloque
   * 
   * Fórmula: gloID = (índice_del_bloque * threads_por_bloque) + thread_local
   * 
   * Ejemplo con imagen 512x512:
   * - Total píxeles: 262,144
   * - Bloques necesarios: ceil(262,144/256) = 1,024
   * - Thread 5 del bloque 3: gloID = 3*256 + 5 = 773 (procesa el píxel 773)
   *============================================================================*/
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Protección contra threads extra
  // Como redondeamos bloques hacia arriba con ceil(), algunos threads
  // pueden tener IDs fuera del rango válido de píxeles
  if (gloID >= w * h) return;

  // Calcular centro de la imagen
  int xCent = w / 2;
  int yCent = h / 2;

  /*============================================================================
   * CONVERSIÓN DE ÍNDICE LINEAL A COORDENADAS CARTESIANAS CENTRADAS
   * 
   * La imagen está almacenada linealmente (row-major):
   * - gloID % w = columna (posición x en la imagen)
   * - gloID / w = fila (posición y en la imagen)
   * 
   * Luego se centra el sistema de coordenadas:
   * - xCoord: columna - xCent (rango: -w/2 a w/2)
   * - yCoord: yCent - fila (invertido para sistema cartesiano)
   * 
   * Visualización:
   * Imagen original:     Sistema centrado:
   * (0,0)--->(w,0)      (-w/2,h/2)---(w/2,h/2)
   *   |                      |           |
   *   v                      |   (0,0)   |
   * (0,h)--->(w,h)      (-w/2,-h/2)---(w/2,-h/2)
   *============================================================================*/
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  // Solo procesar píxeles "encendidos" (blancos)
  if (pic[gloID] > 0)
    {
      // Para cada posible ángulo theta
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          // Calcular distancia r usando valores precalculados de cos y sin
          // r = x*cos(θ) + y*sin(θ)
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          
          // Convertir r a índice del bin correspondiente
          int rIdx = (r + rMax) / rScale;
          
          /*====================================================================
           * USO DE OPERACIÓN ATÓMICA
           * 
           * ¿Por qué atomicAdd si cada thread procesa un píxel diferente?
           * 
           * Respuesta: Múltiples píxeles pueden votar por la MISMA línea.
           * Diferentes píxeles con diferentes coordenadas pueden pertenecer
           * a la misma línea (mismo par r,θ), causando una race condition
           * al incrementar el mismo elemento del acumulador.
           * 
           * Sin atomicAdd, tendríamos perdida de votos por escrituras
           * concurrentes no sincronizadas.
           *====================================================================*/
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

//*****************************************************************
// FUNCIÓN PRINCIPAL
//*****************************************************************
int main (int argc, char **argv)
{
  // Verificar argumentos
  if (argc < 2) {
    printf("Uso: %s <imagen.pgm>\n", argv[0]);
    return 1;
  }

  int i;

  // Cargar imagen PGM de entrada
  PGMImage inImg (argv[1]);
  printf("Imagen cargada: %dx%d píxeles\n", inImg.x_dim, inImg.y_dim);

  int *cpuht;  // Acumulador CPU (para verificación)
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  // ============================================================================
  // ALLOCACIÓN DE MEMORIA PARA TABLAS DE SENOS Y COSENOS EN GPU
  // ============================================================================
  float* d_Cos;
  float* d_Sin;
  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // ============================================================================
  // EJECUCIÓN CPU (REFERENCIA)
  // ============================================================================
  printf("Ejecutando versión CPU...\n");
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);
  printf("CPU completado.\n");

  // ============================================================================
  // PRECÁLCULO DE VALORES TRIGONOMÉTRICOS
  // Se calculan una sola vez para todos los ángulos posibles
  // ============================================================================
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  // Parámetros de la transformada
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // Copiar tablas trigonométricas al GPU
  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

  // ============================================================================
  // PREPARACIÓN DE DATOS PARA GPU
  // ============================================================================
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels;
  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  // Allocar memoria en GPU
  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  
  // Copiar imagen al GPU
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  
  // Inicializar acumulador en GPU a ceros
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // ============================================================================
  // CONFIGURACIÓN Y MEDICIÓN DE TIEMPO CON CUDA EVENTS
  // ============================================================================
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Configuración del kernel: 1 thread por píxel
  int blockNum = ceil(w * h / 256.0);  // Bloques necesarios (redondeado hacia arriba)
  
  printf("\nConfiguración del kernel:\n");
  printf("- Bloques: %d\n", blockNum);
  printf("- Threads por bloque: 256\n");
  printf("- Total threads: %d\n", blockNum * 256);
  printf("- Total píxeles: %d\n", w * h);

  // ============================================================================
  // EJECUCIÓN DEL KERNEL GPU
  // ============================================================================
  printf("\nEjecutando kernel GPU (Memoria Global)...\n");
  
  cudaEventRecord(start);
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
  cudaEventRecord(stop);
  
  cudaEventSynchronize(stop);

  // Calcular tiempo transcurrido
  float kernel_time_ms = 0.0f;
  cudaEventElapsedTime(&kernel_time_ms, start, stop);
  printf("Tiempo de ejecución del kernel: %.4f ms\n", kernel_time_ms);

  // ============================================================================
  // OBTENER RESULTADOS Y VERIFICAR CORRECTITUD
  // ============================================================================
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // Comparar resultados CPU vs GPU
  int errors = 0;
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i]) {
      if (errors < 5) // Mostrar solo los primeros 5 errores
        printf ("Error en posición %i: CPU=%i, GPU=%i\n", i, cpuht[i], h_hough[i]);
      errors++;
    }
  }
  
  if (errors == 0)
    printf("✓ Verificación exitosa: CPU y GPU producen resultados idénticos.\n");
  else
    printf("✗ Se encontraron %d diferencias entre CPU y GPU.\n", errors);

  // ============================================================================
  // DETECCIÓN Y VISUALIZACIÓN DE LÍNEAS
  // ============================================================================
  std::vector<imgutils::Line> lines = imgutils::detect_lines(h_hough, degreeBins, rBins, rMax, 30);
  printf("\nLíneas detectadas: %zu\n", lines.size());

  // Cargar imagen original en RGB para dibujar líneas
  imgutils::Image *img = imgutils::load_pgm_to_rgb(argv[1], w, h);

  if (img) {
    // Dibujar líneas detectadas
    imgutils::draw_lines(img, lines, rMax, rBins);

    // Guardar imagen resultante
    imgutils::save_image_png("hough_result_global.png", img);
    printf("Imagen con líneas guardada: hough_result_global.png\n");

    // Guardar tiempos en CSV
    imgutils::save_timing_csv("timing_global.csv", kernel_time_ms, w, h, lines.size());

    delete img;
  }

  // ============================================================================
  // LIBERACIÓN DE MEMORIA (PUNTO 1b DEL PROYECTO)
  // ============================================================================
  printf("\nLiberando memoria...\n");
  
  // Liberar eventos CUDA
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  // Liberar memoria del host
  free(pcCos);
  free(pcSin);
  delete[] cpuht;  // Usar delete[] porque se creó con new[]
  free(h_hough);
  
  // Liberar memoria del device
  cudaFree(d_Cos);
  cudaFree(d_Sin);
  cudaFree(d_in);
  cudaFree(d_hough);
  
  // Reset del dispositivo CUDA
  cudaDeviceReset();

  printf("✓ Programa completado exitosamente.\n");
  return 0;
}