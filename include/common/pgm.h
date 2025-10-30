// pgm.h - Helpers para leer/escribir imágenes PGM (placeholder)

#ifndef PGM_H
#define PGM_H

#include <string>
#include <vector>

namespace pgm {

// Lee un PGM en escala de grises (P5 o P2) y devuelve ancho, alto y datos (row-major)
bool read_pgm(const std::string& path, int& width, int& height, std::vector<unsigned char>& data);

// Escribe un PGM en escala de grises (P5)
bool write_pgm(const std::string& path, int width, int height, const std::vector<unsigned char>& data);

} // namespace pgm

#endif // PGM_H
