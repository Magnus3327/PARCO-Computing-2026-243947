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
#include <cassert>
#include <cstdlib>

// Project headers 
#include "ResultsManager/ResultsManager.h"
#include "MTX/MTXReader.h"
#include "CSR/CSRMatrix.h"
#include "Utils/Utils.h"
#include "GhostManager/GhostManager.h"

using namespace std;
using namespace mtx;
using namespace utils;

/**
 * Creates a custom MPI derived datatype for the Entry (COO) structure.
 * This allows sending (row, col, value) as a single contiguous block.
 */
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

/**
 * Parses command-line arguments for matrix file path or generation parameters.
 */
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
    opts.iterations = 1;
    if (argc < 2) throw runtime_error("Usage: ./distributedSpMV -M=<file> OR -VM=r;c;d [-I=<iters>]");
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg.rfind("-M=", 0) == 0) {
            opts.filepath = arg.substr(3);
            opts.generateMatrix = false;
            matrixSpecified = true;
        } else if (arg.rfind("-VM=", 0) == 0) {
            stringstream ss(arg.substr(4));
            string token;
            vector<string> values;
            while (getline(ss, token, ';')) if(!token.empty()) values.push_back(token);
            if (values.size() != 3) throw runtime_error("Invalid -VM format. Expected rows;cols;density");
            opts.rows = stoi(values[0]);
            opts.cols = stoi(values[1]);
            opts.density = stod(values[2]);
            opts.generateMatrix = true;
            matrixSpecified = true;
        } else if (arg.rfind("-I=",0)==0) {
            opts.iterations = stoi(arg.substr(3));
        }
    }
    if (!matrixSpecified) throw runtime_error("Matrix source not specified");
    return opts;
}

/**
 * Distributes matrix entries using a 1D cyclic partitioning on rows.
 * Rank 0 splits the matrix and sends chunks to other processes.
 */
void distributeMatrix(const vector<Entry>& allEntries, vector<Entry>& localEntries, int rank, int size, MPI_Datatype entryType) {
    if(rank==0) {
        vector<vector<Entry>> buckets(size);
        for(const auto& e: allEntries){
            int owner = e.row % size;
            Entry local = e;
            local.row = (e.row - owner)/size; // Map to local row index
            buckets[owner].push_back(local);
        }
        for(int p=0; p<size; ++p){
            int count = static_cast<int>(buckets[p].size());
            if(p==0) localEntries = std::move(buckets[0]);
            else {
                MPI_Send(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                if(count>0) MPI_Send(buckets[p].data(), count, entryType, p, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        int count;
        MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(count>0){
            localEntries.resize(count);
            MPI_Recv(localEntries.data(), count, entryType, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

/**
 * Distributes the input vector x using 1D cyclic partitioning.
 * Returns a unique_ptr to the local portion of the vector.
 */
unique_ptr<double[]> distributeVector(int rank, int size, int matrixCols, int& localCols) {
    localCols = matrixCols / size + (rank < (matrixCols % size) ? 1 : 0);
    unique_ptr<double[]> xLocal(new double[max(1, localCols)]);
    if (rank == 0) {
        unique_ptr<double[]> xFull(generateRandomVector(matrixCols, -1000.0, 1000.0));
        for (int p = 0; p < size; ++p) {
            int pCols = matrixCols / size + (p < (matrixCols % size) ? 1 : 0);
            if (pCols == 0) continue;
            vector<double> sendBuf(pCols);
            for (int j = 0; j < pCols; ++j) sendBuf[j] = xFull[p + j * size];
            if (p == 0) for (int j = 0; j < pCols; ++j) xLocal[j] = sendBuf[j];
            else MPI_Send(sendBuf.data(), pCols, MPI_DOUBLE, p, 2, MPI_COMM_WORLD);
        }
    } else if (localCols > 0) {
        MPI_Recv(xLocal.get(), localCols, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return xLocal;
}

/**
 * Distributed SpMV Kernel: Computes y = A * x.
 * Uses pre-indexed offsets to access local or ghost values directly (LUT strategy).
 */
void SpMVDistributed(const CSRMatrix& localCSR, const double* xLocal, const double* xGhost, const vector<int>& xIndex, const vector<char>& isGhost, double* y) {
    for(int i=0; i<localCSR.getRows(); ++i){
        double sum = 0.0;
        for(int j=localCSR.getIndexPointers(i); j<localCSR.getIndexPointers(i+1); ++j){
            // Indirect access through pre-computed LUT offsets
            sum += localCSR.getData(j) * (isGhost[j] ? xGhost[xIndex[j]] : xLocal[xIndex[j]]);
        }
        y[i] = sum;
    }
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ResultsManager rm;
    CLIOptions opts;
    CSRMatrix localCSR;
    vector<Entry> allEntries, localEntries;
    unique_ptr<double[]> x, yLocal;
    vector<int> xIndex;
    vector<char> isGhost;
    GhostManager gm(rank, size); // Custom module to handle remote dependencies

    int iterations=0, matrixRows=0, matrixCols=0, localCols=0;
    MPI_Datatype entryType = createEntryType();
    double globalTime=0.0, time;

    try{
        if(rank==0){
            opts = parseCLI(argc, argv);
            iterations = opts.iterations;
            allEntries = opts.generateMatrix ? generateMatrixEntries(opts.rows, opts.cols, opts.density) : readMTX(opts.filepath);
            
            // Determine dimensions based on input matrix
            matrixRows = 0; matrixCols = 0;
            for(const auto& e: allEntries) {
                if(e.row+1 > matrixRows) matrixRows = e.row+1;
                if(e.col+1 > matrixCols) matrixCols = e.col+1;
            }
            rm.setMatrixInfo(opts.filepath.empty()?"Generated":opts.filepath, opts.generateMatrix, matrixRows, matrixCols, allEntries.size(), opts.density);
            rm.setMPIInfo(size);
        }

        // --- Broadcast Problem metadata ---
        time = MPI_Wtime();
        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // --- Data Distribution ---
        distributeMatrix(allEntries, localEntries, rank, size, entryType);
        localCSR.buildFromEntries(localEntries);
        x = distributeVector(rank, size, matrixCols, localCols);

        // --- NNZ Statistics Collection ---
        size_t localNNZ = localCSR.getNNZ(); 
        size_t minNNZ, maxNNZ, sumNNZ;
        MPI_Reduce(&localNNZ, &minNNZ, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localNNZ, &maxNNZ, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localNNZ, &sumNNZ, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if(rank == 0) rm.setNNZStats(minNNZ, static_cast<double>(sumNNZ)/size, maxNNZ);

        // Discovery phase: Identify off-rank dependencies
        gm.search(localCSR);

        time = (MPI_Wtime() - time) * 1e3;
        if(rank==0) rm.setSetupDuration(time);

        // --- Communication Phase: Exchange Ghost Elements ---
        time = MPI_Wtime();
        gm.exchange(x.get(), localCols);
        time = (MPI_Wtime() - time) * 1e3;
        MPI_Reduce(&time, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(rank==0) rm.setCommunicationDuration(globalTime);

        // --- Pre-indexing (LUT Preparation) ---
        time = MPI_Wtime();
        gm.buildPreIndex(localCSR, xIndex, isGhost, localCols);
        time = (MPI_Wtime() - time) * 1e3;
        MPI_Reduce(&time, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(rank==0) rm.addSetupDuration(globalTime);

        // --- Ghost Statistics Collection ---
        size_t localG = (size > 1) ? gm.getGhostCount() : 0;
        size_t minG, maxG, sumG;
        MPI_Reduce(&localG, &minG, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localG, &maxG, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localG, &sumG, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            rm.setGhostStats(minG, (size > 1 ? (double)sumG/size : 0), maxG, sumG);
        }
                
        // --- Output Allocation ---
        int localRows = matrixRows / size + (rank < (matrixRows % size) ? 1 : 0);
        yLocal = make_unique<double[]>(max(1, localRows));

        // --- Main Kernel Execution ---
        for(int iter=-1; iter<iterations; iter++){
            time = MPI_Wtime();
            // getGhostPtr() provides direct pointer access for maximum kernel performance
            SpMVDistributed(localCSR, x.get(), gm.getGhostPtr(), xIndex, isGhost, yLocal.get());
            time = (MPI_Wtime()-time)*1e3;
            
            MPI_Reduce(&time, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if(iter>=0 && rank==0) rm.addKernelDuration(globalTime);
            else if(iter==-1 && rank==0) rm.setWarmupDuration(globalTime);
        }

        if(rank==0) { 
            rm.computeMetricsBcast(); 
            cout << rm.toJSON() << endl; 
        }

    } catch(const exception& e){
        if(rank==0){ rm.addError(e.what()); cout << rm.toJSON() << endl; }
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Type_free(&entryType);
    MPI_Finalize();
    return 0;
}