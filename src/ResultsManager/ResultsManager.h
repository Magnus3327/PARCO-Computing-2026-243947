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
    size_t totalGhostEntries = 0; // Totale aggregato di tutti i rank

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
    void setMatrixInfo(const string& name, bool generated,
                       size_t rows, size_t cols, size_t nnz, double density);

    void setMPIInfo(int procs);

    void setSetupDuration(double ms);
    void setWarmupDuration(double ms);

    void addKernelDuration(double ms);
    void setCommunicationDuration(double ms);  

    void setGhostInfo(size_t gEntries) { totalGhostEntries = gEntries; }

    // Metrics
    void computeMetrics();

    // Output
    void addError(const string& msg);
    string toJSON() const;

    // Utility
    void clear();
};

#endif // RESULTSMANAGER_H