#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <vector>
#include <string>

namespace imgutils {

    // Estructura para representar una línea detectada
    struct Line {
        float theta;  // ángulo en radianes
        float r;      // distancia
        int weight;   // peso en el acumulador
    };

    // Estructura para imagen RGB
    struct Image {
        unsigned char *data;
        int width;
        int height;
        int channels;  // 1 para escala de grises, 3 para RGB

        Image(int w, int h, int ch) : width(w), height(h), channels(ch) {
            data = new unsigned char[w * h * ch];
        }

        ~Image() {
            delete[] data;
        }
    };

    // Cargar imagen PGM y convertir a RGB
    Image* load_pgm_to_rgb(const char* filename, int width, int height);

    // Detectar líneas en el acumulador de Hough
    std::vector<Line> detect_lines(int *h_hough, int degreeBins, int rBins,
                                    float rMax, int threshold);

    // Dibujar líneas en la imagen original
    void draw_lines(Image *img, const std::vector<Line> &lines, float rMax, int rBins);

    // Función auxiliar para dibujar línea (Bresenham) - declaración forward
    // void draw_line_bresenham(Image *img, int x0, int y0, int x1, int y1,
    //                          unsigned char r, unsigned char g, unsigned char b);

    // Guardar imagen en formato PNG/JPG (usando stb_image_write)
    bool save_image_png(const char *filename, Image *img);

    // Guardar tiempos en CSV
    bool save_timing_csv(const char *filename, float kernel_time_ms,
                         int image_width, int image_height, int num_lines);

} // namespace imgutils

#endif // IMAGE_UTILS_H
