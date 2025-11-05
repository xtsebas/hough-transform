/*
 ============================================================================
 Version       : 2.0
 Last modified : Noviembre 2025
 License       : Released under the GNU GPL 3.0
 Description   : Implementación de Transformada de Hough con Memoria Constante
                 Optimización: Tablas trigonométricas en memoria constante
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
const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

/*============================================================================
 * MEMORIA CONSTANTE (PUNTO 5a DEL PROYECTO)
 * 
 * Declaración de arrays en memoria constante del dispositivo.
 * - Tamaño máximo: 64KB total
 * - Acceso: Read-only desde kernels
 * - Beneficio: Caché dedicado de 8KB por SM
 * - Ideal para: Datos pequeños accedidos frecuentemente por todos los threads
 * 
 * En nuestro caso: 90 floats * 4 bytes * 2 arrays = 720 bytes (<<64KB)
 *============================================================================*/
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

//*****************************************************************
// IMPLEMENTACIÓN CPU (SIN CAMBIOS)
//*****************************************************************
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * degreeBins];
  memset (*acc, 0, sizeof (int) * rBins * degreeBins);
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
      {
        int idx = j * w + i;
        if (pic[idx] > 0)
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;
            float theta = 0;
            for (int tIdx = 0; tIdx < degreeBins; tIdx++)
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++;
                theta += radInc;
              }
          }
      }
}

/*============================================================================
 * KERNEL GPU CON MEMORIA CONSTANTE (PUNTO 5b DEL PROYECTO)
 * 
 * Cambios respecto a la versión con memoria global:
 * 1. NO recibe d_Cos y d_Sin como parámetros (son globales en mem constante)
 * 2. Accede directamente a d_Cos[tIdx] y d_Sin[tIdx]
 * 3. Beneficio: Mejor uso de caché, broadcast automático a threads del warp
 *============================================================================*/
__global__ void GPU_HoughTranConst(unsigned char *pic, int w, int h, int *acc, 
                                    float rMax, float rScale)
{
  // Cálculo del ID global del thread
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h) return;

  int xCent = w / 2;
  int yCent = h / 2;

  // Conversión a coordenadas cartesianas centradas
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          /*==================================================================
           * USO DE MEMORIA CONSTANTE
           * 
           * d_Cos[tIdx] y d_Sin[tIdx] ahora vienen de memoria constante.
           * 
           * Ventajas:
           * 1. Caché dedicado (8KB por SM)
           * 2. Broadcast: cuando todos los threads de un warp acceden al
           *    mismo índice, se hace una sola lectura
           * 3. Reduce tráfico de memoria global
           * 4. Ideal para tablas de lookup pequeñas como esta
           *==================================================================*/
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          
          // Validar límites antes de escribir
          if (rIdx >= 0 && rIdx < rBins) {
            atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
          }
        }
    }
}

//*****************************************************************
// FUNCIÓN PRINCIPAL
//*****************************************************************
int main (int argc, char **argv)
{
  if (argc < 2) {
    printf("Uso: %s <imagen.pgm>\n", argv[0]);
    return 1;
  }

  int i;
  PGMImage inImg (argv[1]);
  printf("=== VERSIÓN CON MEMORIA CONSTANTE ===\n");
  printf("Imagen cargada: %dx%d píxeles\n", inImg.x_dim, inImg.y_dim);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  // ============================================================================
  // EJECUCIÓN CPU
  // ============================================================================
  printf("Ejecutando versión CPU...\n");
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);
  printf("CPU completado.\n");

  // ============================================================================
  // PREPARACIÓN DE VALORES TRIGONOMÉTRICOS (PUNTO 5c DEL PROYECTO)
  // ============================================================================
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  
  printf("\nPrecalculando valores trigonométricos...\n");
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  /*============================================================================
   * TRANSFERENCIA A MEMORIA CONSTANTE (PUNTO 5c DEL PROYECTO)
   * 
   * cudaMemcpyToSymbol vs cudaMemcpy:
   * - cudaMemcpy: para memoria global del device
   * - cudaMemcpyToSymbol: para memoria constante o símbolos globales
   * 
   * Sintaxis: cudaMemcpyToSymbol(symbol, src, size, offset, kind)
   *============================================================================*/
  printf("Copiando tablas trigonométricas a memoria constante...\n");
  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

  // Parámetros de la transformada
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // ============================================================================
  // PREPARACIÓN DE DATOS
  // ============================================================================
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels;
  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // ============================================================================
  // MEDICIÓN DE TIEMPO - MÚLTIPLES EJECUCIONES PARA BITÁCORA
  // ============================================================================
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int blockNum = ceil(w * h / 256.0);
  
  printf("\nConfiguración del kernel:\n");
  printf("- Bloques: %d\n", blockNum);
  printf("- Threads por bloque: 256\n");
  printf("- Memoria constante usada: %zu bytes\n", 2 * sizeof(float) * degreeBins);
  
  const int numMeasurements = 10;
  float times[numMeasurements];
  
  printf("\n=== Ejecutando %d mediciones ===\n", numMeasurements);
  
  for (int m = 0; m < numMeasurements; m++)
  {
    // Limpiar acumulador para cada medición
    cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);
    
    cudaEventRecord(start);
    // Nota: NO pasamos d_Cos y d_Sin como parámetros
    GPU_HoughTranConst<<< blockNum, 256 >>>(d_in, w, h, d_hough, rMax, rScale);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[m], start, stop);
    printf("  Medición %2d: %.4f ms\n", m + 1, times[m]);
  }

  // ============================================================================
  // ESTADÍSTICAS
  // ============================================================================
  float sum = 0, minTime = times[0], maxTime = times[0];
  for (int m = 0; m < numMeasurements; m++) {
    sum += times[m];
    if (times[m] < minTime) minTime = times[m];
    if (times[m] > maxTime) maxTime = times[m];
  }
  float avgTime = sum / numMeasurements;
  
  printf("\n=== ESTADÍSTICAS MEMORIA CONSTANTE ===\n");
  printf("Tiempo promedio: %.4f ms\n", avgTime);
  printf("Tiempo mínimo:   %.4f ms\n", minTime);
  printf("Tiempo máximo:   %.4f ms\n", maxTime);

  // ============================================================================
  // VERIFICACIÓN DE RESULTADOS
  // ============================================================================
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  int errors = 0;
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i]) {
      if (errors < 5)
        printf("Error en posición %i: CPU=%i, GPU=%i\n", i, cpuht[i], h_hough[i]);
      errors++;
    }
  }
  
  if (errors == 0)
    printf("✓ Verificación exitosa: Resultados idénticos CPU vs GPU.\n");
  else
    printf("✗ Se encontraron %d diferencias.\n", errors);

  // ============================================================================
  // DETECCIÓN Y VISUALIZACIÓN
  // ============================================================================
  std::vector<imgutils::Line> lines = imgutils::detect_lines(h_hough, degreeBins, rBins, rMax, 30);
  printf("\nLíneas detectadas: %zu\n", lines.size());

  imgutils::Image *img = imgutils::load_pgm_to_rgb(argv[1], w, h);
  if (img) {
    imgutils::draw_lines(img, lines, rMax, rBins);
    imgutils::save_image_png("hough_result_constant.ppm", img);
    printf("Imagen guardada: hough_result_constant.ppm\n");
    
    // Guardar CSV con tiempo promedio
    imgutils::save_timing_csv("timing_constant.csv", avgTime, w, h, lines.size());
    delete img;
  }

  // ============================================================================
  // COMPARACIÓN CON VERSIÓN GLOBAL
  // ============================================================================
  printf("\n=== COMPARACIÓN DE RENDIMIENTO ===\n");
  printf("Memoria Global:    8.10 ms (referencia anterior)\n");
  printf("Memoria Constante: %.2f ms\n", avgTime);
  float speedup = 8.10f / avgTime;
  printf("Speedup: %.2fx\n", speedup);

  // ============================================================================
  // LIBERACIÓN DE MEMORIA
  // ============================================================================
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(pcCos);
  free(pcSin);
  delete[] cpuht;
  free(h_hough);
  cudaFree(d_in);
  cudaFree(d_hough);
  cudaDeviceReset();

  printf("\n✓ Programa completado exitosamente.\n");
  return 0;
}