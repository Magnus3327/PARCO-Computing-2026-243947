#ifndef GHOST_MANAGER_H
#define GHOST_MANAGER_H

#include <mpi.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <numeric>

// My custom csr library
#include "CSR/CSRMatrix.h"

using namespace std;

class GhostManager {
private:
    int rank, size;
    vector<vector<int>> neededCols; 
    vector<int> sendCounts;         
    vector<int> recvCounts;         
    unique_ptr<double[]> xGhost;
    unordered_map<int, int> ghostMap;

public:
    GhostManager(int rank, int size);
    
    // Check dependecies
    void search(const CSRMatrix& localCSR);
    
    // Exchange ghost values
    void exchange(const double* xLocal, int localCols);
    
    // Mapping phase: creates a Look-Up Table (LUT) in xIndex to allow
    //O(1) direct access during the SpMV kernel.
    void buildPreIndex(const CSRMatrix& localCSR, vector<int>& xIndex, vector<char>& isGhost, int localCols);

    // Getter using ptr to boost performance
    const double* getGhostPtr() const { return xGhost.get(); }
    size_t getGhostCount() const { return ghostMap.size(); }
};

#endif