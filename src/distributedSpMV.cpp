/*
    SPMV - MPI Version

    Distributed Sparse Matrix-Vector Multiplication (SpMV) using MPI.

    WORKFLOW
    --------
    1. Parse CLI arguments (rank 0).
    2. Rank 0 loads or generates the sparse matrix and input vector.
    3. Matrix entries are distributed using 1D cyclic partitioning.
    4. Each rank builds its local CSR matrix.
    5. Warm-up SpMV iteration (not included in statistics).
    6. N timed SpMV iterations.
    7. Rank 0 outputs results in JSON format.

    CLI ARGUMENTS
    -------------
      -M=<path>       Matrix Market file
      -VM=r;c;d       Generate matrix (rows;cols;density)
      -I=<int>        Timed iterations

    COMPILATION
    -----------
      mpic++ -O3 spmv_mpi.cpp -o spmv_mpi
*/

#include <mpi.h>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <algorithm>

// Project headers
#include "ResultsManager/ResultsManager.h"
#include "MTX/MTXReader.h"
#include "CSR/CSRMatrix.h"
#include "Utils/Utils.h"

using namespace std;
using namespace mtx;
using namespace utils;

/* ============================================================
   MPI DERIVED DATATYPE FOR COO ENTRY
   ============================================================ */

MPI_Datatype createEntryType() {
    MPI_Datatype type;

    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};
    MPI_Aint offsets[3];

    offsets[0] = offsetof(Entry, row);
    offsets[1] = offsetof(Entry, col);
    offsets[2] = offsetof(Entry, value);

    MPI_Type_create_struct(3, blocklengths, offsets, types, &type);
    MPI_Type_commit(&type);

    return type;
}

/* ============================================================
   CLI PARSING (RANK 0 ONLY)
   ============================================================ */

struct CLIOptions {
    int iterations = 0;

    string filepath;
    bool generateMatrix = false;

    int rows = 0;
    int cols = 0;
    double density = 0.0;
};

CLIOptions parseCLI(int argc, char* argv[]) {
    CLIOptions opts;
    bool matrixSpecified = false;
    opts.iterations = 1; // Default a 1 se non specificato

    if (argc < 2)
        throw runtime_error("Missing CLI arguments. Usage: ./distributedSpMV -M=<file> OR -VM=r;c;d [-I=<iters>]");

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        // Caso Matrice da File Market
        if (arg.rfind("-M=", 0) == 0) {
            if (matrixSpecified) throw runtime_error("Matrix source already specified");
            if (arg.length() <= 3) throw runtime_error("Filepath for -M is empty");
            
            opts.filepath = arg.substr(3);
            opts.generateMatrix = false;
            matrixSpecified = true;
        }
        // Caso Matrice Generata (Synthetic)
        else if (arg.rfind("-VM=", 0) == 0) {
            if (matrixSpecified) throw runtime_error("Matrix source already specified");
            if (arg.length() <= 4) throw runtime_error("-VM parameters are empty");

            stringstream ss(arg.substr(4));
            string token;
            vector<string> values;

            while (getline(ss, token, ';')) {
                if (!token.empty()) values.push_back(token);
            }

            if (values.size() != 3)
                throw runtime_error("Invalid -VM format. Expected rows;cols;density (e.g., -VM=100;100;0.1)");

            try {
                opts.rows = stoi(values[0]);
                opts.cols = stoi(values[1]);
                opts.density = stod(values[2]);
            } catch (...) {
                throw runtime_error("Invalid numeric values in -VM");
            }

            // Validazione logica dei dati
            if (opts.rows <= 0 || opts.cols <= 0) throw runtime_error("Dimensions must be positive");
            if (opts.density <= 0.0 || opts.density > 1.0) throw runtime_error("Density must be in (0, 1]");

            opts.generateMatrix = true;
            matrixSpecified = true;
        }
        // Caso Iterazioni
        else if (arg.rfind("-I=", 0) == 0) {
            try {
                opts.iterations = stoi(arg.substr(3));
                if (opts.iterations <= 0) opts.iterations = 1; // Protezione contro valori negativi
            } catch (...) {
                throw runtime_error("Invalid iteration count in -I");
            }
        }
        else {
            throw runtime_error("Unknown argument: " + arg);
        }
    }

    if (!matrixSpecified)
        throw runtime_error("Matrix source (-M or -VM) not specified");

    return opts;
}

/* ============================================================
   MATRIX DISTRIBUTION (1D CYCLIC PARTITIONING)
   ============================================================ */

void distributeMatrix(const vector<Entry>& allEntries,
                      vector<Entry>& localEntries,
                      int rank,
                      int size,
                      MPI_Datatype entryType) {

    if (rank == 0) {
        vector<vector<Entry>> buckets(size);

        for (const auto& e : allEntries) {
            int owner = e.row % size;

            Entry local = e;
            local.row = (e.row - owner) / size; // global → local row index

            buckets[owner].push_back(local);
        }

        for (int p = 0; p < size; ++p) {
            int count = static_cast<int>(buckets[p].size());

            if (p == 0) {
                localEntries = std::move(buckets[0]);
            } else {
                MPI_Send(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(buckets[p].data(), count, entryType, p, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        int count;
        MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localEntries.resize(count);
        MPI_Recv(localEntries.data(), count, entryType, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Ogni rank deve riordinare le proprie entry locali prima di costruire il CSR
    sort(localEntries.begin(), localEntries.end(), [](const Entry& a, const Entry& b) {
        if (a.row != b.row) return a.row < b.row;
        return a.col < b.col;
    });
}

/* ============================================================
   DISTRIBUTED SpMV KERNEL
   ============================================================ */

void SpMV_Distributed(const CSRMatrix& localCSR,
                         const double* x, double* y,
                         double& duration_ms) {

    double t0 = MPI_Wtime();

    for (int i = 0; i < localCSR.getRows(); ++i) {
        double sum = 0.0;
        for (int j = localCSR.getIndexPointers(i);
             j < localCSR.getIndexPointers(i + 1); ++j) {
            sum += localCSR.getData(j) * x[localCSR.getIndices(j)];
        }
        y[i] = sum;
    }

    double t1 = MPI_Wtime();
    duration_ms = (t1 - t0) * 1e3;
}

/* ============================================================
   WARM-UP (NOT TIMED)
   ============================================================ */

double warmUp(const CSRMatrix& localCSR, const double* x) {

    // Ensure all processes start together, i don't care wasting time here, because it's just warm-up
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int i = 0; i < localCSR.getRows(); ++i) {
        double sum = 0.0;
        for (int j = localCSR.getIndexPointers(i);
             j < localCSR.getIndexPointers(i + 1); ++j) {
            sum += localCSR.getData(j) * x[localCSR.getIndices(j)];
        }
        volatile double sink = sum;
        (void)sink;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    return (t1 - t0) * 1e3;
}

/* ============================================================
   MAIN
   ============================================================ */

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ResultsManager rm;
    CLIOptions opts;

    vector<Entry> allEntries;
    vector<Entry> localEntries;
    int iterations = 0;
    int globalCols = 0;

    unique_ptr<double[]> x;
    unique_ptr<double[]> y;

    MPI_Datatype entryType = createEntryType();

    try {

        if (rank == 0) {
            opts = parseCLI(argc, argv);
            iterations = opts.iterations;

            if (opts.generateMatrix) {
                allEntries = generateMatrixEntries(opts.rows, opts.cols, opts.density);
            } else {
                allEntries = readMTX(opts.filepath);
                // Calcolo dimensioni reali
                opts.rows = 0;
                opts.cols = 0;
                for (const auto& e : allEntries) {
                    if (e.row + 1 > opts.rows) opts.rows = e.row + 1;
                    if (e.col + 1 > opts.cols) opts.cols = e.col + 1;
                }
            }

            globalCols = opts.cols;
            
            x.reset(generateRandomVector(globalCols, -1000, 1000));

            rm.setMatrixInfo(
                opts.filepath.empty() ? "Generated" : opts.filepath,
                opts.generateMatrix,
                opts.rows,
                opts.cols,
                allEntries.size(),
                opts.density
            );
            rm.setMPIInfo(size); 
        }

        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        MPI_Bcast(&globalCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0)
            x = make_unique<double[]>(globalCols);

        MPI_Bcast(x.get(), globalCols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        distributeMatrix(allEntries, localEntries, rank, size, entryType);

        CSRMatrix localCSR;
        localCSR.buildFromEntries(localEntries);

        double localTime = 0.0, globalTime = 0.0;

        localTime = warmUp(localCSR, x.get());

        MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE,
                   MPI_MAX, 0, MPI_COMM_WORLD);

        y = make_unique<double[]>(localCSR.getRows());

        for (int it = 0; it < iterations; it++) {

            SpMV_Distributed(localCSR, x.get(), y.get(), localTime);

            MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE,
                       MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank == 0)
                rm.addIterationDuration(globalTime);

            // y is intentionally not gathered: result correctness is not the focus here
        }

        if (rank == 0) {
            rm.computeAllMetrics();
            cout << rm.toJSON() << endl;
        }
    }
    catch (const exception& e) {
        if (rank == 0) {
            rm.addError(e.what());
            cout << rm.toJSON() << endl;
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Type_free(&entryType);
    MPI_Finalize();
    return 0;
}