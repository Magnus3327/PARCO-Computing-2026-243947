#include <iostream>
#include <memory>   // unique_ptr
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdlib>  // getenv
#include <chrono>   // high_resolution_clock

// include custom libraries
#include "CSR/CSRMatrix.h"
#include "MTX/MTXReader.h"
#include "BenchmarkResult/BenchmarkResult.h"
#include "Utils/Utils.h"

using namespace std;
using namespace utils;
using namespace mtx;
using namespace chrono;

// -----------------------
// Sequential SpMV
// -----------------------
double* SpMV(const CSRMatrix& csr, const double* x, double& duration) {
    double* y = new double[csr.getRows()];
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < csr.getRows(); i++) {
        double sum = 0.0;
        for (int j = csr.getIndexPointers(i); j < csr.getIndexPointers(i+1); j++)
            sum += csr.getData(j) * x[csr.getIndices(j)];
        y[i] = sum;
    }

    auto end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::nanoseconds>(end - start).count() / 1e6; // ms
    return y;
}

// -----------------------
// CLI Options
// -----------------------
struct CLIOptions {
    string filePath;
    int iterations = 1;
};

// -----------------------
// Parse CLI
// -----------------------
CLIOptions parseCLI(int argc, char* argv[], BenchmarkResult& benchmarkResult) {
    if (argc < 2) {
        benchmarkResult.addError(string(argv[0]) + " requires matrix_path [-I=iterations]");
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
            benchmarkResult.addError("Unknown argument: '" + arg + "'");
            throw runtime_error("Unknown argument");
        }
    }

    return opts;
}

// -----------------------
// Main
// -----------------------
int main(int argc, char* argv[]) {
    BenchmarkResult benchmarkResult;

    try {
        CLIOptions opts = parseCLI(argc, argv, benchmarkResult);

        // Load matrix
        CSRMatrix csr;
        vector<Entry> entries = readMTX(opts.filePath);
        csr.buildFromEntries(entries);

        // Generate input vector
        unique_ptr<double[]> inputVector(generateRandomVector(csr.getCols(), -1000.0, 1000.0));
        unique_ptr<double[]> outputVector(nullptr);

        // Warm-up
        double duration = 0.0;
        outputVector.reset(SpMV(csr, inputVector.get(), duration));
        duration = 0.0;

        // Timed iterations
        for (int i = 0; i < opts.iterations; ++i) {
            outputVector.reset(SpMV(csr, inputVector.get(), duration));
            benchmarkResult.addResult(csr, duration, opts.filePath);
        }

        cout << benchmarkResult.toJSON() << endl;
    }
    catch (const exception& e) {
        benchmarkResult.addError(string("Fatal error: ") + e.what());
        cout << benchmarkResult.toJSON() << endl;
        return 1;
    }

    return 0;
}