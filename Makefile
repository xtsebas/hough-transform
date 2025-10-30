# Compiladores
NVCC = nvcc
CXX = g++

# Flags
NVCC_FLAGS = -O2 -arch=sm_70
CXX_FLAGS = -O2 -std=c++11 -fPIC
INCLUDE = -I./include

# Directorios
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Archivos
CUDA_SRCS = $(SRC_DIR)/houghBase.cu
CXX_SRCS = $(SRC_DIR)/image_utils.cpp
CUDA_OBJS = $(OBJ_DIR)/houghBase.o
CXX_OBJS = $(OBJ_DIR)/image_utils.o
TARGET = $(BIN_DIR)/hough_transform

# Crear directorios si no existen
$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR))

# Regla default
all: $(TARGET)

# Vincular ejecutable
$(TARGET): $(CUDA_OBJS) $(CXX_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compilar archivos CUDA
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE) -c -o $@ $<

# Compilar archivos C++
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE) -c -o $@ $<

# Limpiar
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
