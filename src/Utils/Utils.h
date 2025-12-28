/*
    Utils.h
    
    This header file it's meant to be a collection of utility functions
    
*/

#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <vector>
#include <unordered_set>
#include "MTX/MTXReader.h"

using namespace std;
using namespace mtx;

namespace utils {

    // Generate a random double vector of given size in [minVal, maxVal]
    // The random engine is initialized only once for consistent randomness
    double* generateRandomVector(int size, double minVal = 0.0, double maxVal = 1.0);

    // Generate SparseMatrix in coo format using MTX::Entry
    vector<Entry> generateMatrixEntries(int rows, int cols, double density);

} // namespace utils

#endif // UTILS_H