/*
 * Ghost Manager.h
 *
 * GhostManager class for handling ghost values in distributed SpMV
 * using 1D cyclic partitioning.
 * 
 * According with main file standard, mpi tag used are:
 *   3 - Setup ghost idx
 *   4 - Ghost values
 */
#ifndef GHOSTMANAGER_H
#define GHOSTMANAGER_H

#include <numeric>
#include <vector>
#include <unordered_map>
#include <memory>
#include "CSR/CSRMatrix.h"
#include <mpi.h>

using namespace std;

/*
 * GhostManager class for handling ghost values in distributed SpMV
 * using 1D cyclic partitioning.
 */
class GhostManager {
private:
    int rank, size;

    vector<int> sendCounts, recvCounts;
    vector<vector<int>> neededCols;
    vector<vector<int>> recvCols;

    vector<int> xIndex;
    vector<char> isGhost;
    int totalGhostCount = 0;

public:
    GhostManager(int rank, int size) : rank(rank), size(size) {}

    // identify and prepare ghost columns
    void setup(const CSRMatrix& localCSR);

    // pre-index local and ghost indices for SpMV
    void preIndex(const CSRMatrix& localCSR);

    // Exchange ghost values between ranks
    unique_ptr<double[]> exchangeGhostValues(const double* xLocal, int localCols);

    // Getters
    const vector<int>& getXIndex() const { return xIndex; }
    const vector<char>& getIsGhost() const { return isGhost; }
    int getTotalGhostCount() const { return totalGhostCount; }
};

#endif