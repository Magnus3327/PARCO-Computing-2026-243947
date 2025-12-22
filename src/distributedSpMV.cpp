/*
    SPMV - MPI Version

    Distributed Sparse Matrix-Vector Multiplication (SpMV) using MPI.

    WORKFLOW
    --------
    1. Parse CLI arguments.
    2. Rank 0 loads or generates the matrix and generates the vector.
    3. Vector and portion of the Matrix entries are distributed across processes.
    4. Each rank pergorms its local csr conversion.
    4. Warm-up SpMV iteration (not timed).
    5. N timed SpMV iterations.
    6. Communication overhead is measured.
    7. Rank 0 outputs results in JSON format.

    CLI ARGUMENTS
    -------------
      -M=<path>       Matrix Market file (required)
      -VM=r;c;d       Generate matrix (rows;cols;density) instead of reading from file
      -I=<int>        Timed iterations

    COMPILATION
    -----------
      mpic++ -O3 spmv_mpi.cpp -o spmv_mpi

    RUNNING
    -----------
      mpirun -np <num_processes> ./spmv_mpi -M=matrix.mtx -I=10
      mpirun -np <num_processes> ./spmv_mpi -VM=1000;1000;0.01 -I=10
*/

#include <mpi.h>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <memory>

// Project headers
#include "ResultsManager.h"
#include "MTX/MTXReader.h"

using namespace std;
using namespace mtx;
using namespace utils;

// derived datatype to be distributed 

// Cli Parsing
struct CLIOptions {
    int iterations = 0;

    // Matrix from file
    string filepath = "";

    // Matrix generation
    bool generateMatrix = false;
    int rows = 0;
    int cols = 0;
    double density = 0.0;
};


CLIOptions parseCLI(int argc, char* argv[], ResultsManager& resultsManager) {
    CLIOptions opts;

    if (argc < 2) {
        resultsManager.addError(
            "Usage: -M=<path> | -VM=rows;cols;density  [-I=iterations]"
        );
        throw runtime_error("Insufficient CLI arguments");
    }

    bool matrixSpecified = false;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        // -------- MATRIX FROM FILE --------
        if (arg.rfind("-M=", 0) == 0) {
            if (matrixSpecified)
                throw runtime_error("Matrix source already specified");

            opts.filepath = arg.substr(3);
            if (opts.filepath.empty())
                throw runtime_error("Empty matrix file path");

            opts.generateMatrix = false;
            matrixSpecified = true;
        }

        // -------- GENERATED MATRIX --------
        else if (arg.rfind("-VM=", 0) == 0) {
            if (matrixSpecified)
                throw runtime_error("Matrix source already specified");

            string params = arg.substr(4);
            stringstream ss(params);
            string token;
            vector<string> values;

            while (getline(ss, token, ';'))
                values.push_back(token);

            if (values.size() != 3)
                throw runtime_error("Invalid -VM format. Use -VM=rows;cols;density");

            opts.rows = stoi(values[0]);
            opts.cols = stoi(values[1]);
            opts.density = stod(values[2]);

            if (opts.rows <= 0 || opts.cols <= 0)
                throw runtime_error("Matrix dimensions must be > 0");

            if (opts.density <= 0.0 || opts.density > 1.0)
                throw runtime_error("Density must be in (0, 1]");

            opts.generateMatrix = true;
            matrixSpecified = true;
        }

        // -------- ITERATIONS --------
        else if (arg.rfind("-I=", 0) == 0) {
            opts.iterations = stoi(arg.substr(3));
            if (opts.iterations <= 0)
                throw runtime_error("Iterations must be > 0");
        }

        // -------- UNKNOWN ARG --------
        else {
            throw runtime_error("Unknown argument: '" + arg + "'");
        }
    }

    // -------- FINAL VALIDATION --------
    if (!matrixSpecified)
        throw runtime_error("You must specify either -M=<path> or -VM=rows;cols;density");

    if (opts.iterations == 0)
        throw runtime_error("Missing required argument: -I=<iterations>");

    return opts;
}

// SpMV function (distributed)
double* SpMV_Distributed(const CSRMatrix& localCSR, const double* x, double& duration_ms) {
    double t_start, t_end;
    const int localRows = localCSR.getRows();
    double* y_local = new double[localRows];

    t_start = MPI_Wtime();

    for (int i = 0; i < localRows; ++i) {
        double sum = 0.0;
        for (int j = localCSR.getIndexPointers(i);j < localCSR.getIndexPointers(i + 1); ++j) {
            sum += localCSR.getData(j) * x[localCSR.getIndices(j)];
        }
        y_local[i] = sum;
    }

    t_end = MPI_Wtime();
    duration_ms = (t_end - t_start) * 1e3;  // seconds → ms

    return y_local;
}

// Warm-up function
double warmUp(const CSRMatrix& localCSR, const double* x) {
    double duration_ms = 0.0;

    // sincronizza TUTTI prima
    MPI_Barrier(MPI_COMM_WORLD);

    volatile double sink = SpMV_Distributed(localCSR, x, duration_ms);
    (void) sink;

    // sincronizza TUTTI dopo
    MPI_Barrier(MPI_COMM_WORLD);

    return duration_ms;
}

void distributeMatrix() {
    // Implementation of matrix distribution logic
}

void gatherResults() {
    // Implementation of results gathering logic
}

// Main
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    ResultsManager resultsManager;
    CLIOptions opts;

    try {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        vector<Entry> localEntries;
        unique_ptr<double[]> x = nullptr;
        unique_ptr<double[]> y = nullptr;
        double duration = 0.0;

        if (rank == 0) {
            resultsManager = ResultsManager();
            opts = parseCLI(argc, argv, resultsManager);
            vector<Entry> allEntries;
            
            // read or generate matrix
            if(opts.generateMatrix) 
                allEntries = generateMatrixEntries(opts.rows, opts.cols, opts.density);
            else 
                allEntries = readMTX(opts.filePath);

            cols = max_element(allEntries.begin(), allEntries.end(),
                [](const Entry& a, const Entry& b) { return a.col < b.col; })->col + 1;
            x = generateRandomVector(cols, -1000.0, 1000.0);
            
            //Broadcast vector x to all ranks
            MPI_Bcast(x.get(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Distribute matrix and vector to other ranks
            distributeMatrix(allEntries, x, size);

        } else {
            // Receive local matrix entries and vector
            localEntries = ...; // Receive local entries
            x = make_unique<double[]>(...); // Allocate space for x
            MPI_Bcast(x.get(), ..., MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Each rank builds its local CSR matrix
        CSRMatrix localCSR;
        localCSR.buildFromEntries(localEntries);

        // Warm-Up and rank 0 logic to store results
        double warmup_local = warmUp(localCSR, x.get());

        double warmup_global = 0.0;
        MPI_Reduce(
            &warmup_local,
            &warmup_global,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0) {
            resultsManager.setWarmupDuration(warmup_global);
        }

        // Each rank performs timed SpMV iterations
        SpMV_Distributed(localEntries, x, y);

        // Gather results at rank 0
        gatherResults(y, rank, size);

        if (rank == 0) {
            // Output results in JSON format
            resultsManager.outputJSON();
        }

    } catch (const exception& e) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            resultsManager.addError(e.what());
            resultsManager.outputJSON();
        }
    }

    MPI_Finalize();
    return 0;
}