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
    3 - Ghost column indices
    4 - Ghost values
    
    WORKFLOW
    --------
    1. Parse CLI arguments (rank 0).
    2. Rank 0 loads or generates the sparse matrix and input vector.
    3. Matrix entries are distributed using 1D cyclic partitioning.
    4. Input vector x is distributed using 1D cyclic partitioning.
    5. Each rank identifies and exchanges ghost values needed for local SpMV.
    6. Perform distributed SpMV for a number of iterations, measuring performance.
    7. Collect and report performance metrics (rank 0).
    8. Finalize MPI.

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
#include <unordered_map>

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

    // Matrix file path
    string filepath;

    // Matrix generation parameters
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
            if (matrixSpecified) 
                throw runtime_error("Matrix source already specified");
            if (arg.length() <= 3) 
                throw runtime_error("Filepath for -M is empty");
            
            opts.filepath = arg.substr(3);
            opts.generateMatrix = false;
            matrixSpecified = true;

        } else if (arg.rfind("-VM=", 0) == 0) {
            if (matrixSpecified) 
                throw runtime_error("Matrix source already specified");
            if (arg.length() <= 4) 
                throw runtime_error("-VM parameters are empty");

            stringstream ss(arg.substr(4));
            string token;
            vector<string> values;

            while (getline(ss, token, ';')) if (!token.empty()) values.push_back(token);

            if (values.size() != 3) 
                throw runtime_error("Invalid -VM format. Expected rows;cols;density (e.g., -VM=100;100;0.1)");

            try {
                opts.rows = stoi(values[0]);
                opts.cols = stoi(values[1]);
                opts.density = stod(values[2]);
            } catch (...) {
                throw runtime_error("Invalid numeric values in -VM");
            }

            if (opts.rows <= 0 || opts.cols <= 0) 
                throw runtime_error("Dimensions must be positive");
            if (opts.density <= 0.0 || opts.density > 1.0) 
                throw runtime_error("Density must be in (0, 1]");

            opts.generateMatrix = true;
            matrixSpecified = true;
        } else if (arg.rfind("-I=", 0) == 0) {
            try {
                opts.iterations = stoi(arg.substr(3));

                if (opts.iterations <= 0) 
                    opts.iterations = 1;
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

// DISTRIBUTE MATRIX USING 1D CYCLIC - BLOCKING VERSION
void distributeMatrix(const vector<Entry>& allEntries, vector<Entry>& localEntries, int rank, int size, MPI_Datatype entryType) {
    if (rank == 0) {
        vector<vector<Entry>> buckets(size);

        for (const auto& e : allEntries) {
            int owner = e.row % size;

            Entry local = e;

            local.row = (e.row - owner) / size; // local row index, global to local
            buckets[owner].push_back(local);
        }

        for (int p = 0; p < size; ++p) {
            int count = static_cast<int>(buckets[p].size());

            if (p == 0) {
                localEntries = std::move(buckets[0]);
            } else {
                // Send count first
                MPI_Send(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

                // Send entries
                MPI_Send(buckets[p].data(), count, entryType, p, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        int count;

        MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        localEntries.resize(count);
        MPI_Recv(localEntries.data(), count, entryType, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// DISTRIBUTE VECTOR X USING 1D CYCLIC - BLOCKING VERSION
void distributeVector(int rank, int size, int matrixCols, unique_ptr<double[]>& xLocal) {
    int localCols = (matrixCols + size - 1 - rank) / size;
    xLocal = make_unique<double[]>(localCols);

    if (rank == 0) {
        double* xFull = generateRandomVector(matrixCols, -1000, 1000);

        for (int p = 0; p < size; ++p) {
            int pCols = (matrixCols + size - 1 - p) / size;

            if (p == 0) {
                for (int j = 0; j < pCols; j++)
                    xLocal[j] = xFull[p + j * size];
            } else {
                // Create derived type for stride
                MPI_Datatype strideType = createVectorStrideType(pCols, size);
                MPI_Send(xFull + p, 1, strideType, p, 2, MPI_COMM_WORLD);
                MPI_Type_free(&strideType);
            }
        }

        delete[] xFull;
    } else {
        MPI_Recv(xLocal.get(), localCols, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void exchangeGhostValues(
int rank, int size, const CSRMatrix& localCSR, const unique_ptr<double[]>& xLocal, unique_ptr<double[]>& xGhost, unordered_map<int,int>& ghostMap) {
    // Identify needed remote columns per rank
    vector<vector<int>> neededCols(size);

    for (int i = 0; i < localCSR.getRows(); ++i) {
        for (int j = localCSR.getIndexPointers(i);
             j < localCSR.getIndexPointers(i + 1); ++j) {

            int col = localCSR.getIndices(j);
            int owner = col % size;

            if (owner != rank)
                neededCols[owner].push_back(col);
        }
    }

    // Deduplicate + sort, per rank. Avoids redundant requests.
    for (int p = 0; p < size; ++p) {
        auto& v = neededCols[p];

        sort(v.begin(), v.end());
        v.erase(unique(v.begin(), v.end()), v.end());
    }

    // Exchange counts, using Alltoall.
    vector<int> sendCounts(size, 0), recvCounts(size, 0);
    for (int p = 0; p < size; ++p)
        sendCounts[p] = static_cast<int>(neededCols[p].size());

    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Exchange column indices
    vector<vector<int>> recvCols(size);
    vector<MPI_Request> reqs;

    for (int p = 0; p < size; ++p) {
        if (recvCounts[p] > 0) {
            recvCols[p].resize(recvCounts[p]); // allocate space
            
            MPI_Request r;
            MPI_Irecv(recvCols[p].data(), recvCounts[p], MPI_INT, p, 3, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }

    for (int p = 0; p < size; ++p) {
        if (sendCounts[p] > 0) {
            MPI_Request r;
            MPI_Isend(neededCols[p].data(), sendCounts[p],MPI_INT, p, 3, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }

    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();

    // Prepare values to send (xLocal → sendVals)
    vector<vector<double>> sendVals(size);

    for (int p = 0; p < size; ++p) {
        if (recvCounts[p] == 0) continue;

        sendVals[p].resize(recvCounts[p]);
        for (int i = 0; i < recvCounts[p]; ++i) {
            int col = recvCols[p][i];
            int localIdx = (col - rank) / size; // local index in xLocal

            if (localIdx < 0)
                throw runtime_error("Invalid localIdx in ghost send");

            sendVals[p][i] = xLocal[localIdx];
        }
    }

    // Exchange ghost values
    vector<vector<double>> recvVals(size);

    for (int p = 0; p < size; ++p) {
        if (sendCounts[p] > 0) {
            recvVals[p].resize(sendCounts[p]);

            MPI_Request r;
            MPI_Irecv(recvVals[p].data(), sendCounts[p], MPI_DOUBLE, p, 4, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }

    for (int p = 0; p < size; ++p) {
        if (recvCounts[p] > 0) {
            MPI_Request r;
            MPI_Isend(sendVals[p].data(), recvCounts[p], MPI_DOUBLE, p, 4, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }

    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    // Build xGhost and mapping (col → xGhost index)
    int ghostCount = 0;
    for (int p = 0; p < size; ++p)
        ghostCount += static_cast<int>(recvVals[p].size());

    xGhost = make_unique<double[]>(ghostCount);
    ghostMap.clear();

    int pos = 0;
    for (int p = 0; p < size; ++p) {
        for (int i = 0; i < recvVals[p].size(); ++i) {
            int col = neededCols[p][i];  
            
            ghostMap[col] = pos;
            xGhost[pos++] = recvVals[p][i];
        }
    }

    if (pos != ghostCount)
        throw runtime_error("Ghost exchange mismatch");
}

// DISTRIBUTED SpMV KERNEL WITH GHOSTS
void SpMVDistributed(const CSRMatrix& localCSR, const double* xLocal, const unique_ptr<double[]>& xGhost, const unordered_map<int,int>& ghostMap, double* y, int rank, int size) {
    for (int i = 0; i < localCSR.getRows(); ++i) {
        
        double sum = 0.0;
        
        for (int j = localCSR.getIndexPointers(i); j < localCSR.getIndexPointers(i + 1); ++j) {
            int col = localCSR.getIndices(j);
            
            if (col % size == rank) {
                int localIdx = (col - rank) / size; // local index in xLocal
                sum += localCSR.getData(j) * xLocal[localIdx];
            } else {
                sum += localCSR.getData(j) * xGhost[ghostMap.find(col)->second]; // using find to avoid double lookup
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

    unique_ptr<double[]> xLocal;
    unique_ptr<double[]> yLocal;

    // ghost management
    // xGhost: values of remote vector elements needed for local computation
    // ghostMap: map from global column -> index in xGhost
    unique_ptr<double[]> xGhost;
    unordered_map<int,int> ghostMap;

    int iterations = 0, matrixRows = 0, matrixCols = 0;
    
    MPI_Datatype entryType = createEntryType();

    double globalTime = 0.0; // for max reduction
    double time; // used for duration measurements

    try {
        if (rank == 0) {
            // Parse CLI
            opts = parseCLI(argc, argv);
            iterations = opts.iterations;

            if (opts.generateMatrix)
                allEntries = generateMatrixEntries(opts.rows, opts.cols, opts.density);
            else 
                allEntries = readMTX(opts.filepath);

            opts.rows = 0; opts.cols = 0;
            for (const auto& e : allEntries) {
                if (e.row + 1 > opts.rows) opts.rows = e.row + 1;
                if (e.col + 1 > opts.cols) opts.cols = e.col + 1;
            }

            rm.setMatrixInfo(opts.filepath.empty() ? "Generated" : opts.filepath, opts.generateMatrix, opts.rows, opts.cols, allEntries.size(), opts.density);
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

        // Distribute matrix entries and build local CSR
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
        
        // Distribute vector x  
        distributeVector(rank, size, matrixCols, xLocal);

        time = (MPI_Wtime() - time) * 1e3; // setup duration
        if(rank==0) rm.setSetupDuration(time);

        // Ghost exchanging
        time = MPI_Wtime();
        exchangeGhostValues(rank, size, localCSR, xLocal, xGhost, ghostMap);
        time = (MPI_Wtime() - time) * 1e3; // communication duration

        MPI_Reduce(&time, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(rank==0) rm.setCommunicationDuration(globalTime);
        
        // Gather ghost statistics
        size_t localGhosts = ghostMap.size();
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
            SpMVDistributed(localCSR, xLocal.get(), xGhost, ghostMap, yLocal.get(), rank, size);
            time = (MPI_Wtime() - time) * 1e3; // local computation duration

            MPI_Reduce(&time, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if(iter>=0) { if(rank==0) rm.addKernelDuration(globalTime);}
            else { if(rank==0) rm.setWarmupDuration(globalTime);}
        }

        if(rank==0) {
            rm.computeMetrics();
            cout << rm.toJSON() << endl;
        }
    }
    catch(const exception& e) {
        if(rank==0) {
            rm.addError(e.what());
            cout << rm.toJSON() << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Type_free(&entryType);
    MPI_Finalize();
    return 0;
}