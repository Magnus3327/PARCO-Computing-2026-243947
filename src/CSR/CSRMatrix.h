#ifndef CSRMATRIX_H
#define CSRMATRIX_H

#include <vector>
#include <string>
#include <stdexcept>
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
    // Costruttore / Distruttore
    CSRMatrix();
    ~CSRMatrix();

    // Data accessors
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getNNZ() const { return nnz; }
    const int getIndexPointers(int i) const { return indexPointers[i]; }
    const int getIndices(int i) const { return indices[i]; }
    const double getData(int i) const { return data[i]; }

    // Build CSR from entries containing row, col, value triplets
    void buildFromEntries(const vector<Entry>& entries);

    // For debugging
    void print() const;

    // Liberare memoria manualmente 
    void clear();
};

#endif // CSRMATRIX_H