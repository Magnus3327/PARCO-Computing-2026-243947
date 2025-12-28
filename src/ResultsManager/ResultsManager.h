/**
 * ResultsManager.h
 * 
 * This header defines the ResultsManager class, which is responsible for collecting,
 * managing, and outputting performance metrics and statistics related to sparse matrix-vector over MPI computations.
 */

#ifndef RESULTSMANAGER_H
#define RESULTSMANAGER_H

#include <string>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <sstream>

using namespace std;

class ResultsManager {
private:
    // Matrix metadata 
    string matrixName;
    bool isGenerated = false;
    size_t rows = 0;
    size_t cols = 0;
    size_t nnz = 0;
    double density = 0.0;

    // MPI environment 
    int mpiProcesses = 1;

    // Rank metrics
    // NNZ per rank
    size_t minNNZ = 0;
    size_t avgNNZ = 0;
    size_t maxNNZ = 0;

    // Ghost entries per rank
    size_t minGhostEntries = 0;
    size_t avgGhostEntries = 0;
    size_t maxGhostEntries = 0;
    size_t totalGhostEntries = 0; // sum over all ranks

    // Timing data (milliseconds) 
    double setupDuration = 0.0;          // matrix + vector distribution
    double warmupDuration = 0.0;         // warm-up iteration
    double communicationDuration = 0.0;  // ghost exchange / collectives 
    vector<double> kernelDurations;      // pure SpMV kernel

    // Computed metrics 
    double kernelDuration90 = 0.0;
    double gflops = 0.0;
    double bandwidthGBps = 0.0;

    // Memory metrics
    size_t totalFlops = 0;
    size_t totalBytesMoved = 0;
    size_t memoryFootprintBytes = 0;

    // Errors 
    vector<string> errors;

    // Helpers
    double percentile90(vector<double> v);

public:
    ResultsManager() = default;

    // Setters
    void setMatrixInfo(const string& name, bool generated, size_t rows, size_t cols, size_t nnz, double density);

    void setMPIInfo(int procs);

    void setSetupDuration(double ms);
    void setWarmupDuration(double ms);

    void addKernelDuration(double ms);
    void setCommunicationDuration(double ms);  

    void setGhostStats(size_t minGhosts, size_t avgGhosts, size_t maxGhosts, size_t sumGhosts);
    void setNNZStats(size_t minNNZ, size_t avgNNZ, size_t maxNNZ);

    // Metrics
    void computeMetrics();

    // Output
    void addError(const string& msg);
    string toJSON() const;

    // Utility
    void clear();
};

#endif // RESULTSMANAGER_H