// image_utils.cpp - Implementaciones helper

#include "../include/image_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>

namespace imgutils {

// Forward declaration
static void draw_line_bresenham(Image *img, int x0, int y0, int x1, int y1,
                                unsigned char r, unsigned char g, unsigned char b);

// Cargar imagen PGM y convertir a RGB
Image* load_pgm_to_rgb(const char* filename, int width, int height) {
    Image *img = new Image(width, height, 3);

    // Abrir archivo PGM
    FILE *file = fopen(filename, "rb");
    if (!file) {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
        delete img;
        return nullptr;
    }

    char magic[3];
    int w, h, maxval;

    // Leer header PGM
    fscanf(file, "%2s %d %d %d", magic, &w, &h, &maxval);
    fgetc(file);  // leer whitespace

    // Leer datos en escala de grises
    unsigned char *gray = new unsigned char[w * h];
    fread(gray, sizeof(unsigned char), w * h, file);
    fclose(file);

    // Convertir de escala de grises a RGB (replicar valor en los 3 canales)
    for (int i = 0; i < w * h; i++) {
        img->data[i * 3 + 0] = gray[i];      // R
        img->data[i * 3 + 1] = gray[i];      // G
        img->data[i * 3 + 2] = gray[i];      // B
    }

    delete[] gray;

    std::cout << "Imagen cargada: " << w << "x" << h << " píxeles" << std::endl;

    return img;
}

// Función auxiliar para dibujar línea (Bresenham)
static void draw_line_bresenham(Image *img, int x0, int y0, int x1, int y1,
                                unsigned char r, unsigned char g, unsigned char b) {
    int width = img->width;
    int height = img->height;

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    int x = x0, y = y0;

    while (true) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int idx = (y * width + x) * 3;
            img->data[idx + 0] = r;
            img->data[idx + 1] = g;
            img->data[idx + 2] = b;
        }

        if (x == x1 && y == y1) break;

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

// Detectar líneas en el acumulador de Hough
std::vector<Line> detect_lines(int *h_hough, int degreeBins, int rBins,
                                float rMax, int threshold) {
    std::vector<Line> lines;

    // Iterar sobre el acumulador y encontrar máximos locales
    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            int weight = h_hough[rIdx * degreeBins + tIdx];

            if (weight >= threshold) {
                // Convertir índices a valores reales
                float r = -rMax + (rIdx + 0.5f) * (2.0f * rMax / rBins);
                float theta = tIdx * (M_PI / degreeBins);  // radianes

                Line line;
                line.r = r;
                line.theta = theta;
                line.weight = weight;

                lines.push_back(line);
            }
        }
    }

    std::cout << "Líneas detectadas: " << lines.size() << std::endl;

    return lines;
}

// Dibujar líneas en la imagen original
void draw_lines(Image *img, const std::vector<Line> &lines, float rMax, int rBins) {
    if (!img || lines.empty()) return;

    int width = img->width;
    int height = img->height;
    int xCent = width / 2;
    int yCent = height / 2;

    // Color para las líneas: rojo brillante (255, 0, 0)
    unsigned char red = 255;
    unsigned char green = 0;
    unsigned char blue = 0;

    for (const auto &line : lines) {
        float theta = line.theta;
        float r = line.r;

        float cos_theta = cos(theta);
        float sin_theta = sin(theta);

        // Encontrar dos puntos que satisfacen r = x*cos(θ) + y*sin(θ)
        // Usamos los bordes de la imagen
        int x0 = -xCent;
        int x1 = width - xCent;
        int y0, y1;

        // Calcular y para los dos valores de x en los bordes
        if (fabs(sin_theta) > 0.001f) {
            // y = (r - x*cos(θ)) / sin(θ)
            y0 = (int)((r - x0 * cos_theta) / sin_theta);
            y1 = (int)((r - x1 * cos_theta) / sin_theta);
        } else {
            // Línea aproximadamente vertical
            y0 = -yCent;
            y1 = height - yCent;
        }

        // Convertir a coordenadas de imagen (desde coordenadas centradas)
        int px0 = 0;
        int py0 = y0 + yCent;
        int px1 = width - 1;
        int py1 = y1 + yCent;

        // Clampear a los límites de la imagen
        if (py0 < 0) {
            py0 = 0;
            px0 = (fabs(cos_theta) > 0.001f) ? (int)(-sin_theta * (y0 - (-yCent)) / cos_theta) : 0;
        }
        if (py0 >= height) {
            py0 = height - 1;
            px0 = (fabs(cos_theta) > 0.001f) ? (int)((r - sin_theta * (py0 - yCent)) / cos_theta) : 0;
        }
        if (py1 < 0) {
            py1 = 0;
            px1 = (fabs(cos_theta) > 0.001f) ? (int)(-sin_theta * (y1 - (-yCent)) / cos_theta) : (width - 1);
        }
        if (py1 >= height) {
            py1 = height - 1;
            px1 = (fabs(cos_theta) > 0.001f) ? (int)((r - sin_theta * (py1 - yCent)) / cos_theta) : (width - 1);
        }

        // Dibujar línea usando algoritmo de Bresenham
        draw_line_bresenham(img, px0, py0, px1, py1, red, green, blue);
    }
}

// Guardar imagen en formato PPM (alternativa simple sin dependencias externas)
bool save_image_png(const char *filename, Image *img) {
    if (!img || !img->data) {
        std::cerr << "Error: Imagen vacía" << std::endl;
        return false;
    }

    // Cambiar extensión a .ppm si es necesario
    std::string output_filename = filename;
    size_t dot_pos = output_filename.rfind('.');
    if (dot_pos != std::string::npos) {
        output_filename.replace(dot_pos, std::string::npos, ".ppm");
    }

    FILE *file = fopen(output_filename.c_str(), "wb");
    if (!file) {
        std::cerr << "Error: No se pudo crear el archivo " << output_filename << std::endl;
        return false;
    }

    // Escribir header PPM (P6 = binary RGB)
    fprintf(file, "P6\n");
    fprintf(file, "%d %d\n", img->width, img->height);
    fprintf(file, "255\n");

    // Escribir datos RGB
    size_t bytes_written = fwrite(img->data, sizeof(unsigned char),
                                   img->width * img->height * img->channels, file);

    fclose(file);

    if (bytes_written == (size_t)(img->width * img->height * img->channels)) {
        std::cout << "Imagen guardada: " << output_filename << std::endl;
        return true;
    } else {
        std::cerr << "Error al guardar imagen: " << output_filename << std::endl;
        return false;
    }
}

// Guardar tiempos en CSV
bool save_timing_csv(const char *filename, float kernel_time_ms,
                     int image_width, int image_height, int num_lines) {
    std::ofstream file(filename, std::ios::app);

    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir " << filename << std::endl;
        return false;
    }

    // Escribir header si el archivo está vacío
    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "timestamp,image_width,image_height,num_lines,kernel_time_ms\n";
    }

    // Escribir datos
    file << "2025-10-29," << image_width << "," << image_height << ","
         << num_lines << "," << kernel_time_ms << "\n";

    file.close();
    std::cout << "Tiempos guardados en CSV: " << filename << std::endl;

    return true;
}

} // namespace imgutils
