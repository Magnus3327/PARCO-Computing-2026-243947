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
    3 - Ghost exchange
    
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
#include <numeric>
#include <cmath>

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
    opts.iterations = 1; // Default at least 1 iteration

    if (argc < 2)
        throw runtime_error("Missing CLI arguments. Usage: ./distributedSpMV -M=<file> OR -VM=r;c;d [-I=<iters>]");

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg.rfind("-M=", 0) == 0) {
            if (matrixSpecified) throw runtime_error("Matrix source already specified");
            if (arg.length() <= 3) throw runtime_error("Filepath for -M is empty");
            opts.filepath = arg.substr(3);
            opts.generateMatrix = false;
            matrixSpecified = true;
        } else if (arg.rfind("-VM=", 0) == 0) {
            if (matrixSpecified) throw runtime_error("Matrix source already specified");
            if (arg.length() <= 4) throw runtime_error("-VM parameters are empty");

            stringstream ss(arg.substr(4));
            string token;
            vector<string> values;
            while (getline(ss, token, ';')) if (!token.empty()) values.push_back(token);
            if (values.size() != 3) throw runtime_error("Invalid -VM format. Expected rows;cols;density (e.g., -VM=100;100;0.1)");

            try {
                opts.rows = stoi(values[0]);
                opts.cols = stoi(values[1]);
                opts.density = stod(values[2]);
            } catch (...) {
                throw runtime_error("Invalid numeric values in -VM");
            }

            if (opts.rows <= 0 || opts.cols <= 0) throw runtime_error("Dimensions must be positive");
            if (opts.density <= 0.0 || opts.density > 1.0) throw runtime_error("Density must be in (0, 1]");

            opts.generateMatrix = true;
            matrixSpecified = true;
        } else if (arg.rfind("-I=", 0) == 0) {
            try {
                opts.iterations = stoi(arg.substr(3));
                if (opts.iterations <= 0) opts.iterations = 1;
            } catch (...) {
                throw runtime_error("Invalid iteration count in -I");
            }
        } else {
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
void distributeMatrix(const vector<Entry>& allEntries, vector<Entry>& localEntries, int rank, int size, MPI_Datatype entryType) {
    if (rank == 0) {
        vector<vector<Entry>> buckets(size);
        for (const auto& e : allEntries) {
            int owner = e.row % size;
            Entry local = e;
            local.row = (e.row - owner) / size;
            buckets[owner].push_back(local);
        }

        vector<MPI_Request> requests(2 * (size - 1));
        int reqIndex = 0;
        for (int p = 0; p < size; ++p) {
            int count = static_cast<int>(buckets[p].size());
            if (p == 0) {
                localEntries = std::move(buckets[0]);
            } else {
                MPI_Isend(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &requests[reqIndex++]);
                MPI_Isend(buckets[p].data(), count, entryType, p, 1, MPI_COMM_WORLD, &requests[reqIndex++]);
            }
        }
        if (!requests.empty()) MPI_Waitall(reqIndex, requests.data(), MPI_STATUSES_IGNORE);
    } else {
        int count;
        MPI_Request reqCount;
        MPI_Irecv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &reqCount);
        MPI_Wait(&reqCount, MPI_STATUS_IGNORE);
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
        double* xFull = generateRandomVector(matrixCols, -1000, 1000);
        MPI_Request* requests = new MPI_Request[size - 1];
        int reqIndex = 0;
        vector<MPI_Datatype> strideTypes(size);

        for (int p = 0; p < size; ++p) {
            int pCols = (matrixCols + size - 1 - p) / size;
            if (p == 0) {
                for (int j = 0; j < pCols; j++) xLocal[j] = xFull[p + j * size];
            } else {
                strideTypes[p] = createVectorStrideType(pCols, size);
                MPI_Isend(xFull + p, 1, strideTypes[p], p, 2, MPI_COMM_WORLD, &requests[reqIndex++]);
            }
        }
        if (size > 1) MPI_Waitall(size - 1, requests, MPI_STATUSES_IGNORE);
        for (int p = 1; p < size; ++p) MPI_Type_free(&strideTypes[p]);
        delete[] requests;
        delete[] xFull;
    } else {
        MPI_Request req;
        MPI_Irecv(xLocal.get(), localCols, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

// EXCHANGE GHOST VALUES
void exchangeGhostValues(int rank, int size, const CSRMatrix& localCSR,
                         const unique_ptr<double[]>& xLocal,
                         unique_ptr<double[]>& xGhost,
                         vector<int>& ghostIndices,
                         vector<int>& ghostGlobalToLocal) {
    ghostIndices.clear();
    unordered_set<int> ghostSet;
    vector<vector<int>> recvColsFromRank(size);
    vector<vector<int>> sendColsToRank(size);

    // 1. Identify non-local columns needed
    for (int i = 0; i < localCSR.getRows(); i++) {
        for (int j = localCSR.getIndexPointers(i); j < localCSR.getIndexPointers(i + 1); ++j) {
            int col = localCSR.getIndices(j);
            int owner = col % size;
            if (owner != rank) {
                recvColsFromRank[owner].push_back(col);
                if (ghostSet.find(col) == ghostSet.end()) {
                    ghostSet.insert(col);
                    ghostIndices.push_back(col);
                }
            }
        }
    }

    // 2. Prepare ghost mapping
    int ghostCount = static_cast<int>(ghostIndices.size());
    xGhost = make_unique<double[]>(ghostCount);

    if (ghostCount > 0) {
        ghostGlobalToLocal.resize( *max_element(ghostIndices.begin(), ghostIndices.end()) + 1, -1);
        for (int i = 0; i < ghostCount; i++)
            ghostGlobalToLocal[ghostIndices[i]] = i;
    }

    // 3. Build send lists: which ranks need which of our local x values
    for (int i = 0; i < ghostIndices.size(); ++i) {
        int col = ghostIndices[i];
        int owner = col % size;
        sendColsToRank[owner].push_back(col);
    }

    // 4. Exchange counts first
    vector<int> sendCounts(size, 0);
    vector<int> recvCounts(size, 0);
    for (int p = 0; p < size; ++p) {
        sendCounts[p] = static_cast<int>(sendColsToRank[p].size());
        MPI_Sendrecv(&sendCounts[p], 1, MPI_INT, p, 0,
                     &recvCounts[p], 1, MPI_INT, p, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 5. Post receives
    vector<MPI_Request> recvRequests;
    vector<vector<double>> recvBuffers(size);
    for (int p = 0; p < size; ++p) {
        if (p == rank || recvCounts[p] == 0) continue;
        recvBuffers[p].resize(recvCounts[p]);
        MPI_Request req;
        MPI_Irecv(recvBuffers[p].data(), recvCounts[p], MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &req);
        recvRequests.push_back(req);
    }

    // 6. Send local values
    for (int p = 0; p < size; ++p) {
        if (p == rank || sendCounts[p] == 0) continue;
        vector<double> sendBuffer(sendCounts[p]);
        for (int i = 0; i < sendCounts[p]; i++) {
            int localIdx = (sendColsToRank[p][i] - rank) / size;
            sendBuffer[i] = xLocal[localIdx];
        }
        MPI_Send(sendBuffer.data(), sendCounts[p], MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
    }

    // 7. Wait for all receives
    if (!recvRequests.empty()) MPI_Waitall(static_cast<int>(recvRequests.size()), recvRequests.data(), MPI_STATUSES_IGNORE);

    // 8. Map received values into xGhost
    int pos = 0;
    for (int p = 0; p < size; ++p) {
        for (double val : recvBuffers[p]) xGhost[pos++] = val;
    }
}

// DISTRIBUTED SpMV KERNEL WITH GHOSTS
void SpMVDistributed(const CSRMatrix& localCSR, const double* xLocal, const unique_ptr<double[]>& xGhost, const vector<int>& ghostGlobalToLocal, double* y, int rank, int size) {
    for (int i = 0; i < localCSR.getRows(); ++i) {
        double sum = 0.0;
        for (int j = localCSR.getIndexPointers(i); j < localCSR.getIndexPointers(i + 1); ++j) {
            int col = localCSR.getIndices(j);
            if (col % size == rank) {
                int localIdx = (col - rank) / size;
                sum += localCSR.getData(j) * xLocal[localIdx];
            } else {
                sum += localCSR.getData(j) * xGhost[ghostGlobalToLocal[col]];
            }
        }
        y[i] = sum;
    }
}

// Main
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ResultsManager rm;
    CLIOptions opts;

    CSRMatrix localCSR;

    vector<Entry> allEntries;
    vector<Entry> localEntries;

    // local data
    unique_ptr<double[]> xLocal;
    unique_ptr<double[]> yLocal;

    // ghost management
    unique_ptr<double[]> xGhost;
    vector<int> ghostIndices;
    vector<int> ghostGlobalToLocal;

    int iterations = 0, matrixRows = 0, matrixCols = 0;
    
    MPI_Datatype entryType = createEntryType();

    double globalTime = 0.0; // for max reduction
    double time; // used for duration measurements

    try {
        if (rank == 0) {
            opts = parseCLI(argc, argv);
            iterations = opts.iterations;
            if (opts.generateMatrix) allEntries = generateMatrixEntries(opts.rows, opts.cols, opts.density);
            else allEntries = readMTX(opts.filepath);

            opts.rows = 0; opts.cols = 0;
            for (const auto& e : allEntries) {
                if (e.row + 1 > opts.rows) opts.rows = e.row + 1;
                if (e.col + 1 > opts.cols) opts.cols = e.col + 1;
            }

            rm.setMatrixInfo(opts.filepath.empty() ? "Generated" : opts.filepath,
                             opts.generateMatrix, opts.rows, opts.cols,
                             allEntries.size(), opts.density);
            rm.setMPIInfo(size);

            iterations = opts.iterations;
            matrixRows = opts.rows;
            matrixCols = opts.cols;
        }
        time = MPI_Wtime(); 

        // Broadcast problem info
        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        distributeMatrix(allEntries, localEntries, rank, size, entryType);
        localCSR.buildFromEntries(localEntries);

        // Gather NNZ statistics per rank
        size_t localNNZ = localCSR.getNNZ();
        size_t minNNZ, maxNNZ, sumNNZ;

        MPI_Reduce(&localNNZ, &minNNZ, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localNNZ, &maxNNZ, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localNNZ, &sumNNZ, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            size_t avgNNZ = static_cast<double>(sumNNZ) / size;
            rm.setNNZStats(minNNZ, avgNNZ, maxNNZ);
        }

        yLocal = make_unique<double[]>(localCSR.getRows());
        distributeVector(rank, size, matrixCols, xLocal);

        time = (MPI_Wtime() - time) * 1e3; // setup duration
        if(rank==0) rm.setSetupDuration(time);

        // Ghost exchanging
        time = MPI_Wtime();
        exchangeGhostValues(rank, size, localCSR, xLocal, xGhost, ghostIndices, ghostGlobalToLocal);
        time = (MPI_Wtime() - time) * 1e3; // communication duration

        MPI_Reduce(&time, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(rank==0) rm.setCommunicationDuration(globalTime);
        
        // Gather ghost statistics
        size_t localGhosts = ghostIndices.size();
        size_t minGhosts, maxGhosts, sumGhosts;

        MPI_Reduce(&localGhosts, &minGhosts, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localGhosts, &maxGhosts, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localGhosts, &sumGhosts, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double avgGhosts = static_cast<double>(sumGhosts) / size;
            rm.setGhostStats(minGhosts, avgGhosts, maxGhosts, sumGhosts);
        }

        for(int iter=-1; iter<iterations; iter++) {
            MPI_Barrier(MPI_COMM_WORLD); // synchronize before computation

            time = MPI_Wtime();
            SpMVDistributed(localCSR, xLocal.get(), xGhost, ghostGlobalToLocal, yLocal.get(), rank, size);
            time = (MPI_Wtime() - time) * 1e3; // local computation duration

            MPI_Reduce(&time, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if(iter>=0) { if(rank==0) rm.addKernelDuration(globalTime);}
            else { if(rank==0) rm.setWarmupDuration(globalTime);}
        }

        if(rank==0) {
            rm.computeMetrics();
            std::cout << rm.toJSON() << std::endl;
        }
    }
    catch(const std::exception& e) {
        if(rank==0) {
            rm.addError(e.what());
            std::cout << rm.toJSON() << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Type_free(&entryType);
    MPI_Finalize();
    return 0;
}