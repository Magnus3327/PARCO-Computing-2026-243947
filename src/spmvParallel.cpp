#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib> // getenv
#include <memory> // unique_ptr

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

    #ifdef _OPENMP
    const char* omp_env = getenv("OMP_NUM_THREADS");
    opts.numThreads = (omp_env) ? max(1, atoi(omp_env)) : omp_get_max_threads();
    #endif

    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        int val;

        if (arg.rfind("-T=", 0) == 0) {
            val = stoi(arg.substr(3));
            if (val <= 0) throw runtime_error("numThreads must be > 0");
            opts.numThreads = val;  // assegna solo se è valido
        } 
        else if (arg.rfind("-S=", 0) == 0) {
            string val = arg.substr(3);
            if (val != "static" && val != "dynamic" && val != "guided") {
                throw runtime_error("Invalid scheduling type. Allowed: static, dynamic, guided");
            }
            opts.schedulingType = val; // assegna solo se valido
        } 
        else if (arg.rfind("-C=", 0) == 0) {
            int val = stoi(arg.substr(3));
            if (val < 0) throw runtime_error("chunkSize must be >= 0");
            opts.chunkSize = val; // assegna solo se valido
        } 
        else if (arg.rfind("-I=", 0) == 0) {
            int val = stoi(arg.substr(3));
            if (val <= 0) throw runtime_error("iterations must be > 0");
            opts.iterations = val; // assegna solo se valido
        } 
        else {
            throw runtime_error("Unknown argument: '" + arg + "'");
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