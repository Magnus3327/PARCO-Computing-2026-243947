/*
    SPMV Parallel

    This program performs Sparse Matrix-Vector Multiplication (SpMV) using OpenMP
    to parallelize the operation. The matrix is stored in Compressed Sparse Row (CSR) format.

    Workflow:
    1. Reads a sparse matrix from a Matrix Market (MTX) file.
    2. Generates a random input vector.
    3. Performs the SpMV operation for a configurable number of iterations.
    4. Measures execution time for each iteration.
    5. Stores results (matrix details, scheduling type, chunk size, number of threads,
       and duration) in a ResultsManager and outputs JSON.

    CLI ARGUMENTS:
    - matrix_path         (mandatory) Path to the MTX file.
    - -T=num_threads      (optional) Number of OpenMP threads. Defaults to max threads or OMP_NUM_THREADS.
    - -S=scheduling       (optional) OpenMP scheduling: static, dynamic, or guided. Default is static.
    - -C=chunkSize        (optional) Chunk size for OpenMP scheduling. Default is 0 (let OpenMP decide).
    - -I=iterations       (optional) Number of SpMV iterations. Default is 1.

    OPTIMIZATIONS:
    - Warm-up phase before timed iterations to avoid measuring thread creation overhead.
    - Use of unique_ptr for automatic memory management of dynamic arrays.
    - Input and output vectors stored as pointers for faster access.
    - CSRMatrix members accessed via inline getters to reduce access time.

    USAGE SUGGESTION:
    Redirect output to a file for easier parsing:
        ./spmvParallel matrix.mtx -T=4 -S=dynamic > output.json

    COMPILATION:
    Requires OpenMP support, e.g.:
        g++ -std=c++11 -O3 -fopenmp spmvParallel.cpp ...
*/

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib> // getenv
#include <memory>  // unique_ptr

#include "CSR/CSRMatrix.h"
#include "MTX/MTXReader.h"
#include "ResultsManager/ResultsManager.h"
#include "Utils/Utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace utils;
using namespace mtx;

// SpMV function (parallel)
double* SpMV(const CSRMatrix& csr, const double* x, double& duration, string schedulingType = "static", int chunksize = 0) 
{
    double* y = new double[csr.getRows()];
    double start = 0.0, end = 0.0;

    #ifdef _OPENMP
    // Set OpenMP scheduling policy
    omp_sched_t schedule;
    if (schedulingType == "static") schedule = omp_sched_static;
    else if (schedulingType == "dynamic") schedule = omp_sched_dynamic;
    else if (schedulingType == "guided") schedule = omp_sched_guided;
    else throw runtime_error("Invalid scheduling type: use static, dynamic, or guided.");

    omp_set_schedule(schedule, chunksize);
    start = omp_get_wtime();
    #endif

    // Parallel row-wise SpMV
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < csr.getRows(); i++) {
        double sum = 0.0;  // accumulate row sum
        for (int j = csr.getIndexPointers(i); j < csr.getIndexPointers(i+1); j++) {
            sum += csr.getData(j) * x[csr.getIndices(j)];
        }
        y[i] = sum;
    }

    #ifdef _OPENMP
    end = omp_get_wtime();
    #endif

    duration = (end - start) * 1e3; // convert seconds to milliseconds
    return y;
}

// CLI parsing
struct CLIOptions {
    string filePath;
    string schedulingType = "static";
    int chunkSize = 0;
    int iterations = 1;
    int numThreads = 1;
};

CLIOptions parseCLI(int argc, char* argv[], ResultsManager& resultsManager) {
    if (argc < 2) {
        resultsManager.addError(string(argv[0]) + " requires matrix_path [-T=num_threads] [-S=scheduling] [-C=chunkSize] [-I=iterations]");
        throw runtime_error("Insufficient CLI arguments");
    }

    CLIOptions opts;
    opts.filePath = argv[1];

    // Default threads from OMP_NUM_THREADS or max available
    #ifdef _OPENMP
    const char* omp_env = getenv("OMP_NUM_THREADS");
    opts.numThreads = (omp_env) ? max(1, atoi(omp_env)) : omp_get_max_threads();
    #endif

    // Parse additional options
    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        int val;

        if (arg.rfind("-T=", 0) == 0) {
            val = stoi(arg.substr(3));
            if (val <= 0) throw runtime_error("numThreads must be > 0");
            opts.numThreads = val;
        } 
        else if (arg.rfind("-S=", 0) == 0) {
            string val = arg.substr(3);
            if (val != "static" && val != "dynamic" && val != "guided")
                throw runtime_error("Invalid scheduling type. Allowed: static, dynamic, guided");
            opts.schedulingType = val;
        } 
        else if (arg.rfind("-C=", 0) == 0) {
            val = stoi(arg.substr(3));
            if (val < 0) throw runtime_error("chunkSize must be >= 0");
            opts.chunkSize = val;
        } 
        else if (arg.rfind("-I=", 0) == 0) {
            val = stoi(arg.substr(3));
            if (val <= 0) throw runtime_error("iterations must be > 0");
            opts.iterations = val;
        } 
        else {
            throw runtime_error("Unknown argument: '" + arg + "'");
        }
    }

    // Cap requested threads to max available
    #ifdef _OPENMP
        int maxThreads = omp_get_max_threads();
        if (opts.numThreads > maxThreads) {
            resultsManager.addError("Requested threads (" + to_string(opts.numThreads) + ") exceeds maximum available (" + to_string(maxThreads) + "). Using max.");
            opts.numThreads = maxThreads;
        }
    #endif

    return opts;
}

// Main
int main(int argc, char* argv[]) {
    ResultsManager resultsManager;

    try {
        CLIOptions opts = parseCLI(argc, argv, resultsManager);

        #ifdef _OPENMP
        omp_set_num_threads(opts.numThreads);
        #endif

        // Load matrix
        CSRMatrix csr;
        vector<Entry> entries = readMTX(opts.filePath);
        csr.buildFromEntries(entries);

        // Generate input vector
        unique_ptr<double[]> inputVector(generateRandomVector(csr.getCols(), -1000.0, 1000.0));
        unique_ptr<double[]> outputVector(nullptr);

        // Warm-up
        double duration = 0.0;
        outputVector.reset(SpMV(csr, inputVector.get(), duration, opts.schedulingType, opts.chunkSize));
        duration = 0.0;

        // Actual Timed iterations
        for (int i = 0; i < opts.iterations; ++i) {
            outputVector.reset(SpMV(csr, inputVector.get(), duration, opts.schedulingType, opts.chunkSize));
            resultsManager.addResult(csr, opts.numThreads, opts.schedulingType, opts.chunkSize, duration, opts.filePath);
        }

        cout << resultsManager.toJSON() << endl;
    }
    catch (const exception& e) {
        resultsManager.addError(string("Fatal error: ") + e.what());
        cout << resultsManager.toJSON() << endl;
        return 1;
    }

    return 0;
}