/*
    SPMV Sequential

    This program performs Sparse Matrix-Vector Multiplication (SpMV) sequentially
    using the Compressed Sparse Row (CSR) format for efficient sparse matrix storage.
    It exports detailed performance metrics in JSON format.

    WORKFLOW
    --------
    1. Reads a sparse matrix from a Matrix Market (.mtx) file.
    2. Converts it into CSR format.
    3. Generates a random input vector.
    4. Executes a warm-up SpMV iteration (not timed) counting bytes moved and FLOPs.
    5. Executes N timed SpMV iterations (N = -I=iterations).
    6. Stores timings and metadata in ResultsManager.
    7. Computes:
        - 90th percentile iteration time
        - FLOP count
        - GFLOPS
        - Memory Bandwidth (GB/s)
        - Arithmetic Intensity (FLOPs/Byte)
    8. Prints a JSON block containing:
        * Matrix metadata
        * Execution scenario (threads=1 in sequential)
        * 90th percentile performance statistics
        * Warm-up time
        * List of all iteration durations
        * Any warnings or errors collected

    CLI ARGUMENTS
    -------------
      matrix_path         Path to the input .mtx matrix (REQUIRED)
      -I=<int>            Number of timed iterations (optional, default 1)

    Automatically handles:
      - Validation of user input
      - Automatic memory management with unique_ptr
      - Accurate timing and real byte/FLOP counting during warm-up

    OUTPUT FORMAT (JSON)
    --------------------
    {
        "matrix": {
            "name": <string>,
            "rows": <int>,
            "cols": <int>,
            "nnz": <int>
        },
        "statistics90": {
            "duration_ms": <double>,
            "FLOPs": <double>,
            "GFLOPS": <double>,
            "Bandwidth_GB/s": <double>,
            "arithmetic_intensity": <double>
        },
        "warmUp_time_ms": <double>,
        "all_iteration_times_ms": [ <double>, ... ],
        "errors": [ <string>, ... ]
    }

    COMPILATION
    -----------
        using makefile:
            make spmvSequential

    NOTE
    ----
    No OpenMP parallelism; single-threaded execution.
    Changing iterations does not require recompilation.
*/


#include <iostream>
#include <memory> 
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono> // high_resolution_clock

// Include project headers
#include "CSR/CSRMatrix.h"
#include "MTX/MTXReader.h"
#include "ResultsManager/ResultsManager.h"
#include "Utils/Utils.h"

using namespace std;
using namespace utils;
using namespace mtx;
using namespace chrono;

// SpMV function (sequential)
double* SpMV(const CSRMatrix& csr, const double* x, double& duration) {
    double* y = new double[csr.getRows()];
    auto start = high_resolution_clock::now();

    for (int i = 0; i < csr.getRows(); i++) {
        double sum = 0.0;
        for (int j = csr.getIndexPointers(i); j < csr.getIndexPointers(i+1); j++) {
            sum += csr.getData(j) * x[csr.getIndices(j)];
        }
        y[i] = sum;
    }

    auto end = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end - start).count() / 1e6; // ms
    return y;
}

// Warm-up function also counting bytes moved and flops 
void warmUp(const CSRMatrix& csr, const double* x, double& duration, size_t& bytesMoved, size_t& flopsMoved) {
    double* y = new double[csr.getRows()];
    auto start = high_resolution_clock::now();

    bytesMoved = 0;
    flopsMoved = 0;

    for (int i = 0; i < csr.getRows(); i++) {
        double sum = 0.0;
        for (int j = csr.getIndexPointers(i); j < csr.getIndexPointers(i + 1); j++) {
            sum += csr.getData(j) * x[csr.getIndices(j)];
            bytesMoved += sizeof(double); // csr data
            bytesMoved += sizeof(int);    // csr indices
            bytesMoved += sizeof(double); // input vector x
            flopsMoved += 2;              // 1 mul + 1 add
        }
        y[i] = sum;
        bytesMoved += sizeof(double); // output vector y
    }

    auto end = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end - start).count() / 1e6; // ms

    delete[] y;
}

// CLI Options
struct CLIOptions {
    string filePath;
    int iterations = 1;
};

CLIOptions parseCLI(int argc, char* argv[], ResultsManager& resultsManager) {
    // Basic argument check
    if (argc < 2) {
        resultsManager.addError(string(argv[0]) + " requires matrix_path [-I=iterations]");
        throw runtime_error("Insufficient CLI arguments");
    }

    CLIOptions opts;
    opts.filePath = argv[1];

    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        if (arg.rfind("-I=", 0) == 0) {
            int it = stoi(arg.substr(3));
            if (it <= 0) throw runtime_error("Iterations must be > 0");
            opts.iterations = it;
        } else {
            resultsManager.addError("Unknown argument: '" + arg + "'");
            throw runtime_error("Unknown argument");
        }
    }

    return opts;
}

int main(int argc, char* argv[]) {
    ResultsManager resultsManager;

    try {
        CLIOptions opts = parseCLI(argc, argv, resultsManager);
        double duration = 0.0;
        size_t bytesMoved = 0, flopsMoved = 0;

        // Load matrix
        CSRMatrix csr;
        vector<Entry> entries = readMTX(opts.filePath);
        csr.buildFromEntries(entries);

        string matrixName = opts.filePath.substr(opts.filePath.find_last_of("/\\") + 1);
        resultsManager.setInformation(&csr, matrixName);

        // Generate input vector
        unique_ptr<double[]> inputVector(generateRandomVector(csr.getCols(), -1000.0, 1000.0));
        unique_ptr<double[]> outputVector(nullptr);

        // Warm-up phase
        warmUp(csr, inputVector.get(), duration, bytesMoved, flopsMoved);
        resultsManager.setWarmupDuration(duration);
        resultsManager.setRealTimeMetrics(bytesMoved, flopsMoved);

        // Timed iterations
        for (int i = 0; i < opts.iterations; i++) {
            outputVector.reset(SpMV(csr, inputVector.get(), duration));
            resultsManager.addIterationDuration(duration);
        }

        // Compute final metrics (percentiles, GFLOPS, bandwidth)
        resultsManager.computeAllMetrics();

        // print JSON output
        cout << resultsManager.toJSON() << endl;
    }
    catch (const exception& e) {
        resultsManager.addError(string("Fatal error: ") + e.what());
        cout << resultsManager.toJSON() << endl;
        return 1;
    }

    return 0;
}
