/*
    SPMV Sequential 

    This program performs Sparse Matrix-Vector Multiplication (SpMV) sequentially 
    in CSR format with warm-up, multiple iterations, detailed performance metrics,
    and JSON output.
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
        for (int j = csr.getIndexPointers(i); j < csr.getIndexPointers(i + 1); j++) {
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
