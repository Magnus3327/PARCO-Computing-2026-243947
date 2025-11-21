/*
    SPMV Parallel

    This program performs Sparse Matrix-Vector Multiplication (SpMV) using OpenMP
    and exports detailed performance metrics in JSON format.

    WORKFLOW
    --------
    1. Reads a sparse matrix from a Matrix Market (.mtx) file.
    2. Converts it into CSR (Compressed Sparse Row) format.
    3. Generates a random input vector.
    4. Executes a warm-up SpMV iteration (not timed).
    5. Executes N timed SpMV iterations (N = -I=iterations).
    6. Stores timings and metadata in ResultsManager.
    7. Computes:
        - 90th percentile iteration time
        - FLOP count
        - GFLOPS
        - Memory Bandwidth (GB/s)
    8. Prints a JSON block containing:
        * Matrix metadata
        * Execution scenario (threads, schedule type, chunk size)
        * 90th percentile performance statistics
        * Warm-up time
        * List of all iteration durations
        * Any warnings or errors collected

    CLI ARGUMENTS
    -------------
      matrix_path         Path to the input .mtx matrix (REQUIRED)
      -T=<int>            Number of OpenMP threads
      -S=<string>         Scheduling: static | dynamic | guided
      -C=<int>            Chunk size for OpenMP scheduling
      -I=<int>            Number of timed iterations

    Automatically handles:
      - Validation of user input
      - Capping threads to system max (with warning in JSON)
      - Runtime OpenMP scheduling configuration
      - Automatic memory management with unique_ptr

    OUTPUT FORMAT (JSON)
    --------------------
    {
        "matrix": {
            "name": <string>,
            "rows": <int>,
            "cols": <int>,
            "nnz": <int>
        },
        "scenario": {
            "threads": <int>,
            "scheduling_type": <string>,
            "chunk_size": <int>
        },
        "statistics90": {
            "duration_ms": <double>,      
            "FLOPs": <double>,               
            "GFLOP/s": <double>,             
            "Bandwidth_GB/s": <double>      
            "arithmetic_intensity": <double> 
        },
        "warmUp_time_ms": <double>,
        "all_iteration_times_ms": [ <double>, ... ],
        "errors": [ <string>, ... ]
    }

    COMPILATION
    -----------
        using makefile:
            make spmvParallel

    NOTE
    ----
    Changing scheduling_type or chunk_size does not require recompilation.
    Everything is controlled via OpenMP runtime settings.
*/

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib> // getenv
#include <memory>  // unique_ptr
#include <cmath>    // fabs

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
        double sum = 0.0;  // accumulate row sum its private to each thread
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

// SpMV warm-up function (parallel) also used to count bytes moved and flops during warm-up, instead of estimating them.
void warmUp(const CSRMatrix& csr, const double* x, double& duration, string schedulingType, int chunksize, size_t& bytesMoved, size_t& flopsMoved) {
    double* y = new double[csr.getRows()];
    double start = 0.0, end = 0.0;

    #ifdef _OPENMP
    omp_sched_t schedule;
    if (schedulingType == "static") schedule = omp_sched_static;
    else if (schedulingType == "dynamic") schedule = omp_sched_dynamic;
    else if (schedulingType == "guided") schedule = omp_sched_guided;
    else throw runtime_error("Invalid scheduling type: use static, dynamic, or guided.");

    omp_set_schedule(schedule, chunksize);
    start = omp_get_wtime();
    #endif

    #pragma omp parallel for schedule(runtime) reduction(+:bytesMoved, flopsMoved)
    for (int i = 0; i < csr.getRows(); i++) {
        double sum = 0.0;
        for (int j = csr.getIndexPointers(i); j < csr.getIndexPointers(i+1); j++) {
            sum += csr.getData(j) * x[csr.getIndices(j)];
            bytesMoved += sizeof(double); // csr data
            bytesMoved += sizeof(int);    // csr indices
            bytesMoved += sizeof(double); // input vector x
            flopsMoved += 2;              // 1 mul + 1 add
        }
        y[i] = sum;
        bytesMoved += sizeof(double); // output vector y
    }

    #ifdef _OPENMP
    end = omp_get_wtime();
    #endif

    delete[] y;

    duration = (end - start) * 1e3; // convert ms
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

    // Cap requested threads to max available, instead of throwing error the execution continues reporting the error
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
        double duration = 0.0;

        #ifdef _OPENMP
        omp_set_num_threads(opts.numThreads);
        #endif

        // Load matrix
        CSRMatrix csr;
        vector<Entry> entries = readMTX(opts.filePath);
        csr.buildFromEntries(entries);

        string matrixName = opts.filePath.substr(opts.filePath.find_last_of("/\\") + 1);
        
        // in order to update the csrMatrix pointer in resultsManager, we need to pass by pointer
        resultsManager.setInformation(&csr, opts.numThreads, opts.schedulingType, opts.chunkSize, matrixName);

        // Generate input vector
        unique_ptr<double[]> inputVector(generateRandomVector(csr.getCols(), -1000.0, 1000.0));
        unique_ptr<double[]> outputVector(nullptr);

        // Warm-up Phase
        size_t bytesMoved = 0, flopsMoved = 0;
        warmUp(csr, inputVector.get(), duration, opts.schedulingType, opts.chunkSize, bytesMoved, flopsMoved);
        resultsManager.setWarmupDuration(duration);
        resultsManager.setRealTimeMetrics(bytesMoved, flopsMoved);

        // Actual Timed iterations
        for (int i = 0; i < opts.iterations; i++) {
            outputVector.reset(SpMV(csr, inputVector.get(), duration, opts.schedulingType, opts.chunkSize));
            resultsManager.addIterationDuration(duration);
        }

        // compute statistics about the runs
        resultsManager.computeAllMetrics();

        cout << resultsManager.toJSON() << endl;
    }
    catch (const exception& e) {
        resultsManager.addError(string("Fatal error: ") + e.what());
        cout << resultsManager.toJSON() << endl;
        return 1;
    }

    return 0;
}