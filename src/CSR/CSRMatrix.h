/*
    CSRMatrix.h
    
    This class represents a sparse matrix in Compressed Sparse Row (CSR) format.
    It provides methods to build the CSR representation from a list of entries in coordinate format.

    Therare no setter methodds because the matrix is meant to be built once from entries.
*/

#ifndef CSRMATRIX_H
#define CSRMATRIX_H

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <algorithm> // per std::max_element
#include "MTX/MTXReader.h"

using namespace std;
using mtx::Entry;

class CSRMatrix {
private:
    int *indexPointers;
    int *indices;
    double *data;
    int rows, cols, nnz;

public:
    CSRMatrix();
    ~CSRMatrix();

    // Data accessors
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getNNZ() const { return nnz; }
    const int getIndexPointers(int i) const { return indexPointers[i]; }
    const int getIndices(int i) const { return indices[i]; }
    const double getData(int i) const { return data[i]; }

    // Build CSR from entries containing <row, col, value> triplets
    void buildFromEntries(const vector<Entry>& entries);

    // For debugging
    void print() const;

    // Free allocated memory and reset the matrix
    void clear();
};

#endif // CSRMATRIX_H