#include "../include/image_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#include <vector>

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

//*****************************************************************
// Kernel GPU con memoria compartida completa
__global__ void GPU_HoughTranShared(unsigned char *pic, int w, int h, int *acc,
                                     float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  // Definir locID como threadIdx.x
  int locID = threadIdx.x;

  // Declarar shared int localAcc[degreeBins * rBins]
  extern __shared__ int localAcc[];

  // Inicializar localAcc a 0 con loop desde locID incrementando por blockDim.x
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
  {
    localAcc[i] = 0;
  }

  // Primer __syncthreads() para asegurar que la inicialización esté completa
  __syncthreads();

  // Calcular ID global del thread
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;

          // Cambiar atomicAdd para usar localAcc en vez de acc
          // Validar límites
          if (rIdx >= 0 && rIdx < rBins)
          {
            atomicAdd(localAcc + (rIdx * degreeBins + tIdx), 1);
          }
        }
    }

  // Segunda __syncthreads() para asegurar que todos los updates en localAcc estén completos
  __syncthreads();

  // Loop desde locID hasta degreeBins*rBins (incremento blockDim.x) que haga atomicAdd de localAcc a acc global
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
  {
    if (localAcc[i] > 0)
    {
      atomicAdd(acc + i, localAcc[i]);
    }
  }
}

//*****************************************************************
// The CPU function returns a pointer to the accummulator
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

//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  PGMImage inImg (argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels;

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // CUDA Events para múltiples mediciones
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // execution configuration: 1-D grid of 1-D blocks
  int blockNum = ceil (w * h / 256.0);

  // Tamaño de memoria compartida en bytes (degreeBins * rBins * sizeof(int))
  int sharedMemSize = degreeBins * rBins * sizeof(int);

  // Array para almacenar 10+ mediciones
  float timings[15];
  int numMeasurements = 15;

  printf("Ejecutando kernel GPU_HoughTranShared con memoria compartida completa...\n");
  printf("Grid: %d bloques, Block: 256 threads\n", blockNum);
  printf("Shared memory size: %d bytes\n", sharedMemSize);
  printf("\nMediciones de tiempo (ms):\n");

  // Capturar 10+ mediciones
  for (int m = 0; m < numMeasurements; m++)
  {
    // Resetear acumulador
    cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

    // Registrar tiempo antes del kernel
    cudaEventRecord(start);
    GPU_HoughTranShared<<<blockNum, 256, sharedMemSize>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
    // Registrar tiempo después del kernel
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcular tiempo transcurrido
    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);
    timings[m] = kernel_time_ms;

    printf("Medición %2d: %.4f ms\n", m + 1, kernel_time_ms);
  }

  // Calcular estadísticas
  float sum = 0.0f;
  float minTime = timings[0];
  float maxTime = timings[0];

  for (int m = 0; m < numMeasurements; m++)
  {
    sum += timings[m];
    if (timings[m] < minTime) minTime = timings[m];
    if (timings[m] > maxTime) maxTime = timings[m];
  }

  float avgTime = sum / numMeasurements;

  printf("\n=== Estadísticas ===\n");
  printf("Tiempo promedio: %.4f ms\n", avgTime);
  printf("Tiempo mínimo:   %.4f ms\n", minTime);
  printf("Tiempo máximo:   %.4f ms\n", maxTime);

  // get results from device (última medición)
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // compare CPU and GPU results
  int mismatchCount = 0;
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
    {
      mismatchCount++;
      if (mismatchCount <= 5) // Mostrar solo los primeros 5 errores
        printf ("Calculation mismatch at : %i (CPU: %i, GPU: %i)\n", i, cpuht[i], h_hough[i]);
    }
  }

  if (mismatchCount > 0)
  {
    printf("Total mismatches: %d\n", mismatchCount);
  }
  else
  {
    printf("✓ CPU y GPU resultados coinciden!\n");
  }

  // Detectar líneas
  std::vector<imgutils::Line> lines = imgutils::detect_lines(h_hough, degreeBins, rBins, rMax, 30);

  // Cargar imagen original en RGB
  imgutils::Image *img = imgutils::load_pgm_to_rgb(argv[1], w, h);

  if (img) {
    // Dibujar líneas detectadas
    imgutils::draw_lines(img, lines, rMax, rBins);

    // Guardar imagen resultante
    imgutils::save_image_png("hough_result_shared.png", img);

    // Guardar tiempos en CSV
    imgutils::save_timing_csv("timing_results_shared.csv", avgTime, w, h, lines.size());

    delete img;
  }

  // Limpiar eventos CUDA
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // TODO clean-up
  free(pcCos);
  free(pcSin);
  free(cpuht);
  free(h_hough);
  cudaFree(d_Cos);
  cudaFree(d_Sin);
  cudaFree(d_in);
  cudaFree(d_hough);

  printf("\n✓ Done!\n");

  return 0;
}
