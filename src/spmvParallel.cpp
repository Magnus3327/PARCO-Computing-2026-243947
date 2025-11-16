#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib> // getenv
#include <memory> // unique_ptr, its used for automatic memory management of dynamic arrays

// include custom libraries
#include "CSR/CSRMatrix.h"
#include "MTX/MTXReader.h"
#include "BenchmarkResult/BenchmarkResult.h"
#include "Utils/Utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace utils;
using namespace mtx;

// -----------------------
// SpMV function
// -----------------------
double* SpMV(const CSRMatrix& csr, const double* x, double& duration, string schedulingType = "static", int chunksize = 0) 
{
    double* y = new double[csr.getRows()];
    double start = 0.0, end = 0.0;

    #ifdef _OPENMP
    omp_sched_t schedule;
    if(schedulingType == "static") schedule = omp_sched_static;
    else if(schedulingType == "dynamic") schedule = omp_sched_dynamic;
    else if(schedulingType == "guided") schedule = omp_sched_guided;
    else throw runtime_error("Invalid scheduling type. Use static, dynamic, or guided.");

    omp_set_schedule(schedule, chunksize);
    start = omp_get_wtime();
    #endif

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < csr.getRows(); i++) {
        double sum = 0.0;
        for (int j = csr.getIndexPointers(i); j < csr.getIndexPointers(i+1); j++) {
            sum += csr.getData(j) * x[csr.getIndices(j)];
        }
        y[i] = sum;
    }

    #ifdef _OPENMP
    end = omp_get_wtime();
    #endif

    duration = (end - start) * 1e3; // milliseconds
    return y;
}

// -----------------------
// Helper: parse CLI args
// -----------------------
struct CLIOptions {
    string filePath;
    string schedulingType = "static";
    int chunkSize = 0;
    int iterations = 1;
    int numThreads = 1;
};

CLIOptions parseCLI(int argc, char* argv[], BenchmarkResult& benchmarkResult) {
    if (argc < 2) {
        benchmarkResult.addError(string(argv[0]) + " requires matrix_path [-T=num_threads] [-S=scheduling] [-C=chunkSize] [-I=iterations]");
        throw runtime_error("Insufficient CLI arguments");
    }

    CLIOptions opts;
    opts.filePath = argv[1];

    // Leggi tutti gli argomenti
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

    // --- Check massimo numero di thread solo dopo ---
    #ifdef _OPENMP
        int maxThreads = omp_get_max_threads();
        if (opts.numThreads > maxThreads) {
            benchmarkResult.addError("Requested threads (" + to_string(opts.numThreads) + ") exceeds maximum available (" + to_string(maxThreads) + "). Using max.");
            opts.numThreads = maxThreads;
        }
    #endif

    return opts;
}

// -----------------------
// Main
// -----------------------
int main(int argc, char* argv[]) {
    BenchmarkResult benchmarkResult;

    try {
        CLIOptions opts = parseCLI(argc, argv, benchmarkResult);

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

        // Timed iterations
        for (int i = 0; i < opts.iterations; ++i) {
            outputVector.reset(SpMV(csr, inputVector.get(), duration, opts.schedulingType, opts.chunkSize));
            benchmarkResult.addResult(csr, opts.numThreads, opts.schedulingType, opts.chunkSize, duration, opts.filePath);
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