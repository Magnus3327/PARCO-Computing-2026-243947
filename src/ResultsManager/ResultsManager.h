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

    // Timing data (milliseconds) 
    double setupDuration = 0.0;              // matrix + vector distribution
    double warmupDuration = 0.0;             // warm-up iteration
    vector<double> kernelDurations;          // pure SpMV kernel
    vector<double> communicationDurations;   // ghost exchange / collectives

    // Computed metrics 
    double avgKernelDuration = 0.0;
    double kernelDuration90 = 0.0;
    double avgCommunicationDuration = 0.0;
    double communicationDuration90 = 0.0;

    size_t totalFlops = 0;
    size_t totalBytesMoved = 0;
    size_t memoryFootprintBytes = 0;

    double gflops = 0.0;
    double bandwidthGBps = 0.0;

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
    void addCommunicationDuration(double ms);

    // Metrics
    void computeMetrics();

    // Output
    void addError(const string& msg);
    string toJSON() const;

    // Utility
    void clear();
};

#endif // RESULTSMANAGER_H