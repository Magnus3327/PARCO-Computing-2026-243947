/*
    ResultsManager - Deliverable 2 (MPI Edition)

    This class collects and exports performance metrics for distributed 
    Sparse Matrix-Vector Multiplication (SpMV) using MPI.

    JSON OUTPUT FORMAT:
    {
        "matrix": { "name", "is_generated", "rows", "cols", "nnz", "density" },
        "execution_env": { "mpi_processes", "nodes", "total_memory_gb", "notes" },
        "benchmarking": { "warmup_time_ms", "iterations", "all_iteration_times_ms" },
        "statistics": { 
            "avg_duration_ms", "p90_duration_ms", "total_flops", 
            "total_gflops", "total_bandwidth_gbps", "mpi_overhead_pct" 
        },
        "errors": []
    }
*/

#ifndef RESULTSMANAGER_H
#define RESULTSMANAGER_H

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

class ResultsManager {
private:
    // Matrix metadata
    string pathmtx = "";
    string matrixName = "";
    bool isGenerated = false;
    size_t rows = 0;
    size_t cols = 0;
    size_t nnz = 0;
    double density = 0.0;

    // MPI Environment
    int mpiProcesses = 1;
    int nodes = 1;
    double totalMemoryGB = 0.0;
    string runNotes;

    // Benchmarking data
    double warmupDuration = 0.0;
    vector<double> iterationDurations;

    // Computed Metrics
    double avgDuration = 0.0;
    double duration90 = 0.0;
    size_t totalFlops = 0;
    size_t totalBytesMoved = 0;
    double gflops = 0.0;
    double bandwidthGBps = 0.0;
    double mpiOverheadPct = 0.0;

    vector<string> errors;

public:
    ResultsManager() = default;
    ~ResultsManager() { clear(); }

    // Setters for matrix and environment
    void setMatrixInfo(const string& name, bool generated, size_t r, size_t c, size_t n, double dens);
    void setMPIInfo(int procs, int nNodes, double mem, const string& notes);

    // Timing and overhead
    void setWarmupDuration(double duration);
    void addIterationDuration(double duration);
    void setMPIOverhead(double commTimeMs); // Total communication time in ms

    // Calculations
    void computeAllMetrics();

    // Utilities
    void addError(const string& msg);
    string toJSON() const;
    void clear();
};

#endif // RESULTSMANAGER_H