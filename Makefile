# Compiler
CXX ?= g++ # use g++ by default, can be overridden by environment variable such as alias or export
CXXFLAGS = -std=c++11 -O3 -I./src

# OpenMP flag
OMPFLAG = -fopenmp

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
CSR_SRC = $(SRC_DIR)/CSR/CSRMatrix.cpp
MTX_SRC = $(SRC_DIR)/MTX/MTXReader.cpp
UTILS_SRC = $(SRC_DIR)/Utils/Utils.cpp
MANAGER_SRC = $(SRC_DIR)/ResultsManager/ResultsManager.cpp

SEQUENTIAL_SRC = $(SRC_DIR)/spmvSequential.cpp
PARALLEL_SRC   = $(SRC_DIR)/spmvParallel.cpp

# Object files
CSR_OBJ   = $(OBJ_DIR)/CSR/CSRMatrix.o
MTX_OBJ   = $(OBJ_DIR)/MTX/MTXReader.o
UTILS_OBJ = $(OBJ_DIR)/Utils/Utils.o
MANAGER_OBJ = $(OBJ_DIR)/ResultsManager/ResultsManager.o

COMMON_OBJS = $(CSR_OBJ) $(MTX_OBJ) $(UTILS_OBJ) $(MANAGER_OBJ)

# Default target
all: help

# Help
help:
	@echo "Usage:"
	@echo "  make sequential   # compile the sequential version"
	@echo "  make parallel     # compile the parallel version with OpenMP"
	@echo "  make clean        # remove all obj and bin files"

# Create obj and bin directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/CSR
	mkdir -p $(OBJ_DIR)/MTX
	mkdir -p $(OBJ_DIR)/Utils
	mkdir -p $(OBJ_DIR)/ResultsManager
	mkdir -p $(BIN_DIR)

# Compile object files
$(CSR_OBJ): $(CSR_SRC) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MTX_OBJ): $(MTX_SRC) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(UTILS_OBJ): $(UTILS_SRC) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MANAGER_OBJ): $(MANAGER_SRC) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Sequential executable
sequential: $(COMMON_OBJS) $(SEQUENTIAL_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_OBJS) $(SEQUENTIAL_SRC) -o $(BIN_DIR)/spmvSequential

# Parallel executable
parallel: $(COMMON_OBJS) $(PARALLEL_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_OBJS) $(PARALLEL_SRC) $(OMPFLAG) -o $(BIN_DIR)/spmvParallel

# Clean all build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)