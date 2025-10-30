// pgm.h - Helpers para leer/escribir imágenes PGM

#ifndef PGM_H
#define PGM_H

#include <string>
#include <vector>
#include <cstdio>
#include <cstring>

// Clase PGMImage para cargar/almacenar imágenes PGM
class PGMImage {
public:
    int x_dim;
    int y_dim;
    unsigned char *pixels;

    PGMImage(const char *filename) {
        FILE *file = fopen(filename, "rb");
        if (!file) {
            fprintf(stderr, "Error: No se pudo abrir %s\n", filename);
            x_dim = 0;
            y_dim = 0;
            pixels = nullptr;
            return;
        }

        char magic[3];
        int maxval;

        // Leer header PGM
        fscanf(file, "%2s", magic);
        fscanf(file, "%d %d %d", &x_dim, &y_dim, &maxval);
        fgetc(file);  // leer whitespace

        // Allocate memory
        pixels = new unsigned char[x_dim * y_dim];

        // Leer datos
        if (magic[1] == '5') {  // P5 (binary)
            fread(pixels, sizeof(unsigned char), x_dim * y_dim, file);
        } else if (magic[1] == '2') {  // P2 (ASCII)
            for (int i = 0; i < x_dim * y_dim; i++) {
                int val;
                fscanf(file, "%d", &val);
                pixels[i] = (unsigned char)val;
            }
        }

        fclose(file);
    }

    ~PGMImage() {
        if (pixels) {
            delete[] pixels;
        }
    }
};

namespace pgm {

// Lee un PGM en escala de grises (P5 o P2) y devuelve ancho, alto y datos (row-major)
bool read_pgm(const std::string& path, int& width, int& height, std::vector<unsigned char>& data);

// Escribe un PGM en escala de grises (P5)
bool write_pgm(const std::string& path, int width, int height, const std::vector<unsigned char>& data);

} // namespace pgm

#endif // PGM_H
