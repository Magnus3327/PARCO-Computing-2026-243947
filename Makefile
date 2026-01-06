# Makefile for Distributed SpMV (MPI)

# MPI Compiler
CXX = mpic++
CXXFLAGS = -O3 -I./src

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Module Source Files
CSR_SRC        = $(SRC_DIR)/CSR/CSRMatrix.cpp
MTX_SRC        = $(SRC_DIR)/MTX/MTXReader.cpp
UTILS_SRC      = $(SRC_DIR)/Utils/Utils.cpp
MANAGER_SRC    = $(SRC_DIR)/ResultsManager/ResultsManager.cpp
GHOST_SRC      = $(SRC_DIR)/GhostManager/GhostManager.cpp

# Main Source Files
MAIN_SRC       = $(SRC_DIR)/distributedSpMV.cpp
MAIN_B_SRC     = $(SRC_DIR)/distributedBcastSpMV.cpp

# Object Files
CSR_OBJ        = $(OBJ_DIR)/CSR/CSRMatrix.o
MTX_OBJ        = $(OBJ_DIR)/MTX/MTXReader.o
UTILS_OBJ      = $(OBJ_DIR)/Utils/Utils.o
MANAGER_OBJ    = $(OBJ_DIR)/ResultsManager/ResultsManager.o
GHOST_OBJ      = $(OBJ_DIR)/GhostManager/GhostManager.o

MAIN_OBJ       = $(OBJ_DIR)/distributedSpMV.o
MAIN_B_OBJ     = $(OBJ_DIR)/distributedBcastSpMV.o

COMMON_OBJS    = $(CSR_OBJ) $(MTX_OBJ) $(UTILS_OBJ) $(MANAGER_OBJ) $(GHOST_OBJ)

# Default target
all: distributed bcast

# Create obj and bin directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/{CSR,MTX,Utils,ResultsManager,GhostManager}
	mkdir -p $(BIN_DIR)

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Distributed executable target
distributed: $(COMMON_OBJS) $(MAIN_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $(BIN_DIR)/distributedSpMV

# Broadcast executable target
bcast: $(COMMON_OBJS) $(MAIN_B_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $(BIN_DIR)/distributedBcastSpMV

# Cleanup
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all distributed bcast clean help

help:
	@echo "Available targets:"
	@echo "  make distributed  # Build distributed SpMV version"
	@echo "  make bcast        # Build broadcast SpMV version"
	@echo "  make clean        # Remove binaries and objects"
	@echo "  make help         # Show this help"