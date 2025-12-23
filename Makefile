# Makefile for Distributed SpMV (MPI)

# Compilatore MPI
CXX = mpic++
CXXFLAGS = -std=c++11 -O3 -I./src

# Directory
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# File Sorgente dei Moduli
CSR_SRC     = $(SRC_DIR)/CSR/CSRMatrix.cpp
MTX_SRC     = $(SRC_DIR)/MTX/MTXReader.cpp
UTILS_SRC   = $(SRC_DIR)/Utils/Utils.cpp
MANAGER_SRC = $(SRC_DIR)/ResultsManager/ResultsManager.cpp

# File Sorgente Principale (visto nello screenshot)
MAIN_SRC    = $(SRC_DIR)/distributedSpMV.cpp

# File Oggetto
CSR_OBJ     = $(OBJ_DIR)/CSR/CSRMatrix.o
MTX_OBJ     = $(OBJ_DIR)/MTX/MTXReader.o
UTILS_OBJ   = $(OBJ_DIR)/Utils/Utils.o
MANAGER_OBJ = $(OBJ_DIR)/ResultsManager/ResultsManager.o
MAIN_OBJ    = $(OBJ_DIR)/distributedSpMV.o

COMMON_OBJS = $(CSR_OBJ) $(MTX_OBJ) $(UTILS_OBJ) $(MANAGER_OBJ)

# Target di default
all: distributed

# Creazione directory obj e bin
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/CSR
	mkdir -p $(OBJ_DIR)/MTX
	mkdir -p $(OBJ_DIR)/Utils
	mkdir -p $(OBJ_DIR)/ResultsManager
	mkdir -p $(BIN_DIR)

# Compilazione degli oggetti (Moduli)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Target per l'eseguibile distribuito
distributed: $(COMMON_OBJS) $(MAIN_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_OBJS) $(MAIN_SRC) -o $(BIN_DIR)/distributedSpMV

# Pulizia
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

help:
	@echo "Comandi disponibili:"
	@echo "  make distributed  # Compila la versione MPI"
	@echo "  make clean        # Rimuove file binari e oggetti"