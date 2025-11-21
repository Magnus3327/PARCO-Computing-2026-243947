/*
    ResultsManager

    This class collects, computes, and exports performance metrics 
    for Sparse Matrix-Vector Multiplication (SpMV) executions.

    It stores:
      - Matrix metadata (rows, cols, nnz, name)
      - Execution scenario (threads, scheduling policy, chunk size)
      - Warm-up duration
      - Per-iteration durations
      - 90th percentile statistics
      - Computed metrics: FLOPs, GFLOPS, Bandwidth

    JSON OUTPUT FORMAT
    ==================

    PARALLEL MODE:
    {
        "matrix": {
            "name": "<matrixName>",
            "rows": <int>,
            "cols": <int>,
            "nnz": <int>
        },
        "scenario": {
            "threads": <int>,
            "scheduling_type": "<static|dynamic|guided>",
            "chunk_size": <int>
        },
        "statistics90": {
            "duration_ms": <double>,         // 90th percentile iteration time
            "FLOPs": <double>,               // total operations 
            "GFLOP/s": <double>,             // FLOP / time
            "Bandwidth_GB/s": <double>       // total bytes / time
            "arithmetic_intensity": <double> // FLOPs / total bytes
        },
        "warmUp_time_ms": <double>,
        "all_iteration_times_ms": [ <double>, <double>, ... ]
    }

    SEQUENTIAL MODE:
    {
        "matrix": {
            "name": "<matrixName>",
            "rows": <int>,
            "cols": <int>,
            "nnz": <int>
        },
        "statistics90": {
            "duration_ms": <double>,
            "FLOP": <double>,
            "GFLOPS": <double>,
            "Bandwidth_GBps": <double>
        },
        "warmUp_time_ms": <double>,
        "all_iteration_times_ms": [ <double>, <double>, ... ]
    }

    Notes:
    - Duration values are in milliseconds.
    - Bandwidth accounts for all bytes read/written by CSR SpMV.
    - 90th percentile is computed after collecting all iteration durations.
*/

#ifndef RESULTSMANAGER_H
#define RESULTSMANAGER_H

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath> // ceil
#include <stdexcept>
#include "CSR/CSRMatrix.h"

using namespace std;

class ResultsManager {
private:
    CSRMatrix* csrMatrix = nullptr; // pointer to the CSR matrix
    int numThreads = 0;
    string schedulingType;
    int chunkSize = 0;
    string matrixName;
    bool sequential = true;

    double warmupDuration = 0.0;
    vector<double> iterationDurations;

    // Metrics
    double duration90 = 0.0;
    size_t flops = 0;
    size_t bytesMoved = 0;
    double gflops = 0.0;
    double bandwidthGBps = 0.0;
    double arithmeticIntensity = 0.0;

    vector<string> errors;

public:
    ResultsManager() = default;
    ~ResultsManager();

    void setInformation(CSRMatrix* csr, int nThreads, const string schedType, int cSize, const string mtxName);   
    void setInformation(CSRMatrix* csr, const string mtxName); 

    // Warm-up
    void setWarmupDuration(double duration);

    // Real-time metrics during an esecution
    void setRealTimeMetrics(size_t byteMoved, size_t flopsMoved);

    // Add iteration durations one by one or set all
    void addIterationDuration(double duration);
    void setIterationDurations(const vector<double>& durations);

    // Compute all metrics (FLOPs, GFLOPS, Bandwidth, 90th percentile)
    void computeAllMetrics();

    // Add error
    void addError(const string& msg);

    // JSON output
    string toJSON() const;

    // Clear all stored data
    void clear();
};

#endif // RESULTSMANAGER_H