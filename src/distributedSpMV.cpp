/*
    SPMV - MPI Version

    Distributed Sparse Matrix-Vector Multiplication (SpMV) using MPI.
    Implements 1D cyclic partitioning for matrix distribution.
    This implementation focuses on performance benchmarking across multiple MPI processes and does not gather or validate the final result vector.

    MPI TAGS
    ---------
    0 - Problem info (iterations, rows, cols)
    1 - Matrix entries
    2 - Vector x segments
    
    WORKFLOW
    --------
    1. Parse CLI arguments (rank 0).
    2. Rank 0 loads or generates the sparse matrix and input vector.
    3. Matrix entries are distributed using 1D cyclic partitioning.
    4. Each rank builds its local CSR matrix.
    5. Warm-up iteration.
    6. N timed SpMV iterations.
    7. Rank 0 outputs results in JSON format.

    CLI ARGUMENTS
    -------------
      -M=<path>       Matrix Market file
      -VM=r;c;d       Generate matrix (rows;cols;density)
      -I=<int>        Timed iterations

    COMPILATION
    -----------
      mpic++ -O3 distributedSpMV.cpp -o bin/distributedSpMV

    RUNNING
    -------
        mpirun -np <procs> bin/distributedSpMV -M=<filepath> -I=10
        mpirun -np <procs> bin/distributedSpMV "-VM=<rows>;<cols>;<density>" -I=10
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

//MPI DERIVED DATATYPE FOR COO ENTRY
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

// MPI DERIVED DATATYPE FOR CYCLIC VECTOR STRIDE
MPI_Datatype createVectorStrideType(int count, int size) {
    MPI_Datatype type;
    // count: number of elements to send
    // blocklength: 1 element per block
    // stride: jump 'size' elements to get to the next one
    MPI_Type_vector(count, 1, size, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

// CLI PARSING (RANK 0 ONLY)
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

        // MTX from file
        if (arg.rfind("-M=", 0) == 0) {
            if (matrixSpecified) throw runtime_error("Matrix source already specified");
            if (arg.length() <= 3) throw runtime_error("Filepath for -M is empty");
            
            opts.filepath = arg.substr(3);
            opts.generateMatrix = false;
            matrixSpecified = true;
        }
        // Generate matrix
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
        // Iterazions
        else if (arg.rfind("-I=", 0) == 0) {
            try {
                opts.iterations = stoi(arg.substr(3));
                if (opts.iterations <= 0) opts.iterations = 1; // Out of bound safe-guard
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

// SORT ENTRIES BY ROW AND COLUMN
void sortEntriesByRowCol(vector<Entry>& entries) {
    sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        if (a.row != b.row) return a.row < b.row;
        return a.col < b.col;
    });
}

// DISTRIBUTE MATRIX USING 1D CYCLIC
void distributeMatrix(const vector<Entry>& allEntries,vector<Entry>& localEntries, int rank, int size, MPI_Datatype entryType) {
    if (rank == 0) {
        // Step 1: create buckets for each rank
        vector<vector<Entry>> buckets(size);
        for (const auto& e : allEntries) {
            int owner = e.row % size;
            Entry local = e;
            local.row = (e.row - owner) / size; // convert global row to local row
            buckets[owner].push_back(local);
        }

        // Step 2: non-blocking send to all ranks != 0
        vector<MPI_Request> requests(2 * (size - 1)); // one for count, one for data
        int reqIndex = 0;

        for (int p = 0; p < size; ++p) {
            int count = static_cast<int>(buckets[p].size());
            if (p == 0) {
                // keep bucket 0 locally
                localEntries = std::move(buckets[0]);
            } else {
                // non-blocking send of count
                MPI_Isend(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &requests[reqIndex++]);
                // non-blocking send of data
                MPI_Isend(buckets[p].data(), count, entryType, p, 1, MPI_COMM_WORLD, &requests[reqIndex++]);
            }
        }

        // Wait for all sends to complete
        if (!requests.empty()) MPI_Waitall(reqIndex, requests.data(), MPI_STATUSES_IGNORE);

    } else {
        // Step 1: non-blocking receive of count
        int count;
        MPI_Request reqCount;
        MPI_Irecv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &reqCount);
        MPI_Wait(&reqCount, MPI_STATUS_IGNORE);

        // Step 2: non-blocking receive of data
        localEntries.resize(count);
        MPI_Request reqData;
        MPI_Irecv(localEntries.data(), count, entryType, 0, 1, MPI_COMM_WORLD, &reqData);
        MPI_Wait(&reqData, MPI_STATUS_IGNORE);
    }
}

// DISTRIBUTE VECTOR X USING 1D CYCLIC
void distributeVector(int rank, int size, int matrixCols, unique_ptr<double[]>& xLocal) {
    int localCols = (matrixCols + size - 1 - rank) / size;
    xLocal = make_unique<double[]>(localCols);

    if (rank == 0) {
        // Generate full vector x
        double* xFull = generateRandomVector(matrixCols, -1000, 1000);

        MPI_Request* requests = new MPI_Request[size - 1];
        int reqIndex = 0;

        // Vector to store types to free them later
        vector<MPI_Datatype> strideTypes(size);

        for (int p = 0; p < size; ++p) {
            int pCols = (matrixCols + size - 1 - p) / size;

            if (p == 0) {
                // Local copy for rank 0: still needs manual copy because types are for communication
                for (int j = 0; j < pCols; j++) {
                    xLocal[j] = xFull[p + j * size];
                }
            } else {
                // Create a specific type for this rank's portion
                strideTypes[p] = createVectorStrideType(pCols, size);
                
                // Send directly from xFull with stride
                MPI_Isend(xFull + p, 1, strideTypes[p], p, 2, MPI_COMM_WORLD, &requests[reqIndex++]);
            }
        }

        // Wait for all non-blocking sends
        if (size > 1) MPI_Waitall(size - 1, requests, MPI_STATUSES_IGNORE);

        // Cleanup
        for (int p = 1; p < size; ++p) MPI_Type_free(&strideTypes[p]);
        delete[] requests;
        delete[] xFull;
    } else {
        // Receive local portion: it arrives contiguous in xLocal
        MPI_Request req;
        MPI_Irecv(xLocal.get(), localCols, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

// DISTRIBUTED SpMV FUNCTION
void SpMV_Distributed(const CSRMatrix& localCSR, const double* x, double* y, double& duration_ms) {
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

// Main
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Rank 0 object to collect cli options and manage results
    ResultsManager rm;
    CLIOptions opts;

    // Local CSR matrix for each rank
    CSRMatrix localCSR;

    vector<Entry> allEntries = nullptr;
    vector<Entry> localEntries = nullptr;

    // use unique_ptr for automatic memory management
    unique_ptr<double[]> xLocal = nullptr;
    unique_ptr<double[]> xGhost = nullptr;
    unique_ptr<double[]> yLocal = nullptr;

    // Global variables needed by all ranks
    int iterations = 0;
    int matrixRows = 0;
    int matrixCols = 0;

    // Create MPI Datatype for Entry struct to be used in communication
    MPI_Datatype entryType = createEntryType();

    double localTime = 0.0, globalTime = 0.0;

    try {
        if (rank == 0) {
            opts = parseCLI(argc, argv);
            iterations = opts.iterations;

            if (opts.generateMatrix) {
                allEntries = generateMatrixEntries(opts.rows, opts.cols, opts.density);
            } else {
                allEntries = readMTX(opts.filepath);
                
                // Infer rows and cols
                opts.rows = 0;
                opts.cols = 0;
                for (const auto& e : allEntries) {
                    if (e.row + 1 > opts.rows) opts.rows = e.row + 1;
                    if (e.col + 1 > opts.cols) opts.cols = e.col + 1;
                }
            }

            rm.setMatrixInfo(
                opts.filepath.empty() ? "Generated" : opts.filepath,
                opts.generateMatrix,
                opts.rows,
                opts.cols,
                allEntries.size(),
                opts.density
            );
            rm.setMPIInfo(size); 

            iterations = opts.iterations;
            matrixRows = opts.rows;
            matrixCols = opts.cols;
        }

        // Broadcast necessary info to all ranks
        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Distribute matrix entries among ranks
        distributeMatrix(allEntries, localEntries, rank, size, entryType);

        localCSR.buildFromEntries(localEntries);

        // Allocate local output vector y
        y = make_unique<double[]>(localCSR.getRows());

        // Distribute vector x among ranks
        distributeVector(rank, size, matrixCols, xLocal);

        // MAIN COMPUTATION LOOP
        for(int iter = -1; iter < iterations; iter++) {
            MPI_Barrier(MPI_COMM_WORLD); // synchronize before each iteration

            // ghost region handling would go here if needed

            // Synchronize ranks to ensure consistent kernel timing
            MPI_Barrier(MPI_COMM_WORLD);
            
            SpMV_Distributed(localCSR, xLocal.get(), yLocal.get(), localTime);

            MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE,
                        MPI_MAX, 0, MPI_COMM_WORLD);

            if (iter >= 0) // skip warm-up and first iteration{
                if(rank == 0 ) rm.addIterationDuration(globalTime);
            else {
                if(rank == 0 ) rm.addWarmUpDuration(globalTime);
            }
        }

        // performance metrics computation and print of results (rank 0)
        if (rank == 0) {
            rm.computeAllMetrics(size);
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