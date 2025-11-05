# Hough-transform

Descripción del Proyecto
------------------------

Implementación paralela de la Transformada de Hough Lineal usando CUDA para detección de líneas rectas en imágenes en blanco y negro. El objetivo es comparar distintas estrategias de uso de memoria en GPU (Global, Global + Constante y Global + Constante + Compartida) y medir su rendimiento.

Estructura de Archivos (nueva)
-----------------------------

- `src/houghBase.cu` — Ejecutable / esqueleto base y utilidades comunes.
- `src/hough_global.cu` — Implementación/driver para la versión que usa memoria global (baseline).
- `src/hough_constant.cu` — Implementación/driver para la versión que usa memoria constante para parámetros constantes.
- `src/hough_shared.cu` — Implementación/driver para la versión que usa memoria compartida para optimizar accesos.
- `src/image_utils.cpp` — Funciones auxiliares para cargar/guardar/dibujar sobre imágenes.
- `include/common/pgm.h` — Helpers para leer/escribir imágenes PGM (P2/P5).
- `include/image_utils.h` — Declaraciones de las utilidades de imagen.
- `input/` — Contiene imágenes de prueba en formato PGM (ej. `test_image.pgm`, `runway.pgm`).
- `output/` — Contiene resultados (ej. `lines_detected_global.png`, `lines_detected_constant.png`, `lines_detected_shared.png`).
- `data/timings.csv` — Archivo con las mediciones de tiempo de ejecución de los diferentes kernels.

Objetivos y contenidos
----------------------

- Implementar la Transformada de Hough lineal en CUDA con al menos tres variantes que usen:
	- Memoria Global (baseline).
	- Memoria Global + Memoria Constante (para tablas/parametros que no cambian).
	- Memoria Global + Memoria Constante + Memoria Compartida (para reducir accesos globales y reutilizar datos entre hilos del mismo bloque).
- Medir tiempos de ejecución usando CUDA events y volcar los resultados en `data/timings.csv` para análisis comparativo.
- Dibujar las líneas detectadas sobre la imagen de entrada y guardar el resultado en `output/`.

Compilación (sugerencia)
------------------------

El `Makefile` incluido es un placeholder; aquí hay una sugerencia mínima para compilar los distintos drivers con `nvcc`:

```makefile
# Ajusta NVCCFLAGS según tu GPU y sistema
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_61 -Iinclude

SRCS = src/houghBase.cu src/hough_global.cu src/hough_constant.cu src/hough_shared.cu src/image_utils.cpp

all: bin/hough_base bin/hough_global bin/hough_constant bin/hough_shared

bin/hough_base: src/houghBase.cu src/image_utils.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ src/houghBase.cu src/image_utils.cpp

bin/hough_global: src/hough_global.cu src/image_utils.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ src/hough_global.cu src/image_utils.cpp

bin/hough_constant: src/hough_constant.cu src/image_utils.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ src/hough_constant.cu src/image_utils.cpp

bin/hough_shared: src/hough_shared.cu src/image_utils.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ src/hough_shared.cu src/image_utils.cpp

clean:
	rm -f bin/*
```

Nota: Cambia `-arch=sm_61` por la arquitectura de tu GPU (por ejemplo `sm_75`, `sm_80`, etc.).

Uso
---

1. Coloca las imágenes PGM en `input/`.
2. Compila con `make` (ajusta el Makefile si lo modificas).
3. Ejecuta el binario correspondiente (por ejemplo `bin/hough_global input/test_image.pgm output/lines_detected_global.png`).
4. Revisa `data/timings.csv` para las mediciones de rendimiento.

Formato recomendado para `data/timings.csv`
---------------------------------------

Ejemplo:

```
kernel,variant,filename,time_ms
Global,baseline,test_image.pgm,123.45
Global+Const,constant,test_image.pgm,98.12
Global+Const+Shared,shared,test_image.pgm,76.33
```

Notas
-----

- Los archivos en `src/` y `include/` son placeholders; reemplaza con tus implementaciones de kernels y utilidades.
- Si quieres, puedo:
	- Generar un `CMakeLists.txt` en vez de Makefile.
	- Implementar un esqueleto de lectura PGM y dibujo de líneas en `image_utils.cpp`.
	- Añadir scripts para ejecutar benchmarks y rellenar `data/timings.csv` automáticamente.

Licencia
--------

Proyecto bajo la licencia que prefieras — añade un archivo `LICENSE` si quieres dejarlo explícito.
---

1. Coloca imágenes en blanco y negro (PNG) en `input/`.
2. Compila el proyecto con `make` o el comando `nvcc` que prefieras.
3. Ejecuta el binario indicando la imagen de entrada y la ruta de salida (según la interfaz que implementes).
4. Revisa `data/timings.csv` para las mediciones de rendimiento.

Formato de `data/timings.csv`
---------------------------

Se espera un CSV con columnas como:

```
kernel,variant,filename,time_ms
Global,baseline,img1.png,123.45
Global+Const,optimized_constants,img1.png,98.12
Global+Const+Shared,optimized_shared,img1.png,76.33
```

Notas
-----

- Los archivos actualmente son placeholders; reemplaza las implementaciones por tu código.
- Si necesitas, puedo añadir un `CMakeLists.txt` o un `Makefile` funcional específico para CUDA y tu sistema.

Licencia
--------

Proyecto bajo la licencia que prefieras — añade un archivo `LICENSE` si quieres dejarlo explícito.


---

## Resumen: Ejecución completa para verificar todo

```bash
# 1. Compilar
make clean && make

# 2. Crear output directory si no existe
mkdir -p output

# 3. Ejecutar versión GLOBAL (Memory Global)
echo "=== MEMORIA GLOBAL ==="
./bin/hough_transform input/test_image.pgm
# Genera: hough_result_global.png, timing_global.csv

# 4. Ejecutar versión SHARED (Memoria Compartida)
echo "=== MEMORIA COMPARTIDA ==="
./bin/hough_shared input/test_image.pgm
# Genera: hough_result_shared.png, timing_results_shared.csv

# 5. Revisar imágenes generadas
ls -lah *.png *.csv

# 6. Revisar archivos de tiempos
cat timing_*.csv

**Archivos finales generados:**
- `hough_result_global.png` - Líneas detectadas con memoria global
- `hough_result_shared.png` - Líneas detectadas con memoria compartida
- `hough_result_constant.ppm` - Líneas detectadas con memoria constante
- `timing_global.csv` - Tiempos ejecución versión global
- `timing_results_shared.csv` - Tiempos ejecución versión compartida
- `timing_constant.csv` - Tiempos ejecución versión constante

---