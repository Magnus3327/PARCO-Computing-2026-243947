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
    // Metadata Matrice
    string matrixName = "";
    bool isGenerated = false;
    size_t rows = 0;
    size_t cols = 0;
    size_t nnz = 0;
    double density = 0.0;

    // Ambiente MPI
    int mpiProcesses = 1;

    // Dati Benchmarking
    double warmupDuration = 0.0;
    vector<double> iterationDurations;

    // Metriche Calcolate
    double avgDuration = 0.0;
    double duration90 = 0.0;
    size_t totalFlops = 0;
    size_t totalBytesMoved = 0;
    double gflops = 0.0;
    double bandwidthGBps = 0.0;

    vector<string> errors;

public:
    ResultsManager() = default;
    
    // Setters
    void setMatrixInfo(const string& name, bool generated, size_t r, size_t c, size_t n, double dens);
    void setMPIInfo(int procs);
    void addIterationDuration(double duration);
    void addWarmUpDuration(double duration);

    // Calcoli
    void computeAllMetrics(size_t mpiProcesses);

    // Output
    void addError(const string& msg);
    string toJSON() const;
    void clear();
};

#endif