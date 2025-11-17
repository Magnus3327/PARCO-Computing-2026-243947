/*
    SPMV Sequential

    This program performs Sparse Matrix-Vector Multiplication (SpMV) sequentially
    using the Compressed Sparse Row (CSR) format for matrix storage. 

    WORKFLOW:
    1. Reads a sparse matrix from a Matrix Market (MTX) file.
    2. Generates a random input vector.
    3. Performs the SpMV operation for a configurable number of iterations.
    4. Measures execution time for each iteration.
    5. Stores results (matrix details and execution duration) in a ResultsManager
       and outputs JSON.

    CLI ARGUMENTS:
    - matrix_path (mandatory) Path to the MTX file.
    - -I=iterations (optional) Number of SpMV iterations. Default is 1.

    OPTIMIZATIONS:
    - Warm-up phase before timed iterations to reduce measurement overhead.
    - Use of unique_ptr for automatic memory management of dynamic arrays.
    - Input and output vectors stored as raw pointers for faster access.
    - CSRMatrix members accessed via inline getters to reduce access time.

    USAGE SUGGESTION:
    Redirect output to a file for easier parsing:
        ./spmvSequential matrix.mtx -I=5 > output.json
*/

#include <iostream>
#include <memory>   // unique_ptr
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>   // high_resolution_clock

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
    auto start = chrono::high_resolution_clock::now();

    // Row-wise multiplication
    for (int i = 0; i < csr.getRows(); i++) {
        double sum = 0.0; // accumulate row sum its private to each thread
        for (int j = csr.getIndexPointers(i); j < csr.getIndexPointers(i+1); j++)
            sum += csr.getData(j) * x[csr.getIndices(j)];
        y[i] = sum;
    }

    auto end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::nanoseconds>(end - start).count() / 1e6; // ms
    return y;
}

// CLI Options
struct CLIOptions {
    string filePath;
    int iterations = 1;
};

// Parse CLI
CLIOptions parseCLI(int argc, char* argv[], ResultsManager& resultsManager) {
    // Check minimum arguments
    if (argc < 2) {
        resultsManager.addError(string(argv[0]) + " requires matrix_path [-I=iterations]");
        throw runtime_error("Insufficient CLI arguments");
    }

    CLIOptions opts;
    opts.filePath = argv[1];

    // Optional iterations argument
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

// Main
int main(int argc, char* argv[]) {
    ResultsManager resultsManager;

    try {
        CLIOptions opts = parseCLI(argc, argv, resultsManager);
        double duration = 0.0;

        // Load matrix
        CSRMatrix csr;
        vector<Entry> entries = readMTX(opts.filePath);
        csr.buildFromEntries(entries);

        // Generate input vector
        unique_ptr<double[]> inputVector(generateRandomVector(csr.getCols(), -1000.0, 1000.0));
        unique_ptr<double[]> outputVector(nullptr);

        // Warm-up, iterations/3 and at least one
        for(int i=0;i<(opts.iterations/3)+1;i++) outputVector.reset(SpMV(csr, inputVector.get(), duration));
        
        duration = 0.0;

        // Actual Timed iterations
        for (int i = 0; i < opts.iterations; i++) {
            outputVector.reset(SpMV(csr, inputVector.get(), duration));
            resultsManager.addResult(csr, duration, opts.filePath);
        }

        // Print final JSON results
        cout << resultsManager.toJSON() << endl;
    }
    catch (const exception& e) {
        resultsManager.addError(string("Fatal error: ") + e.what());

        // Print JSON with errors
        cout << resultsManager.toJSON() << endl;
        return 1;
    }

    return 0;
}