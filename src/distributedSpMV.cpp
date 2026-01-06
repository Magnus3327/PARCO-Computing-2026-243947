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
    3 - Setup ghost idx
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

    COMPILATION USING MAKEFILE
    -----------
      make distributed

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
    opts.iterations = 1;

    if (argc < 2)
        throw runtime_error("Missing CLI arguments. Usage: ./distributedSpMV -M=<file> OR -VM=r;c;d [-I=<iters>]");

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg.rfind("-M=", 0) == 0) { // MATRIX FILE
            if (matrixSpecified) throw runtime_error("Matrix source already specified");
            if (arg.length() <= 3) throw runtime_error("Filepath for -M is empty");
            
            opts.filepath = arg.substr(3);
            opts.generateMatrix = false;
            matrixSpecified = true;

        } else if (arg.rfind("-VM=", 0) == 0) { // MATRIX GENERATION
            if (matrixSpecified) throw runtime_error("Matrix source already specified");
            if (arg.length() <= 4) throw runtime_error("-VM parameters are empty");

            stringstream ss(arg.substr(4));
            string token;
            vector<string> values;
            
            while (getline(ss, token, ';')) if(!token.empty()) values.push_back(token);

            if (values.size() != 3) throw runtime_error("Invalid -VM format. Expected rows;cols;density");

            try {
                opts.rows = stoi(values[0]);
                opts.cols = stoi(values[1]);
                opts.density = stod(values[2]);
            } catch (...) { throw runtime_error("Invalid numeric values in -VM"); }

            if (opts.rows <=0 || opts.cols <=0) throw runtime_error("Dimensions must be positive");
            if (opts.density <=0.0 || opts.density >1.0) throw runtime_error("Density must be in (0,1]");

            opts.generateMatrix = true;
            matrixSpecified = true;

        } else if (arg.rfind("-I=",0)==0) { // iterations
            try { opts.iterations = stoi(arg.substr(3)); if(opts.iterations<=0) opts.iterations=1; }
            catch (...) { throw runtime_error("Invalid iteration count in -I"); }

        } else throw runtime_error("Unknown argument: "+arg);
    }

    if (!matrixSpecified) throw runtime_error("Matrix source (-M or -VM) not specified");
    return opts;
}

// DISTRIBUTE MATRIX USING 1D CYCLIC - Using blocking sends/receives
void distributeMatrix(const vector<Entry>& allEntries, vector<Entry>& localEntries, int rank, int size, MPI_Datatype entryType) {
    if(rank==0) {
        // Bucket entries by owner
        vector<vector<Entry>> buckets(size);

        for(const auto& e: allEntries){
            int owner = e.row % size;
            Entry local = e;
            local.row = (e.row - owner)/size; // local row index
            buckets[owner].push_back(local);
        }

        for(int p=0; p<size; ++p){
            int count = static_cast<int>(buckets[p].size());

            if(p==0) localEntries = std::move(buckets[0]);
            else{
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

// DISTRIBUTE VECTOR X USING 1D CYCLIC - Using blocking sends/receives
/**
 * Distributes the input vector x using 1D cyclic partitioning.
 * Each process rank owns indices j such that (j % size == rank).
 */
unique_ptr<double[]> distributeVector(int rank, int size, int matrixCols, int& localCols) {
    localCols = matrixCols / size + (rank < (matrixCols % size) ? 1 : 0);
    
    // Allocate memory for the local portion of vector x
    unique_ptr<double[]> xLocal(new double[max(1, localCols)]);

    if (rank == 0) {
        // Rank 0 generates the full vector x
        unique_ptr<double[]> xFull(generateRandomVector(matrixCols, -1000.0, 1000.0));

        for (int p = 0; p < size; ++p) {
            // Calculate how many elements belong to process p 
            int pCols = matrixCols / size + (p < (matrixCols % size) ? 1 : 0);
            if (pCols == 0) continue;

            // Extract cyclic elements: x[p], x[p + size], x[p + 2*size]...
            vector<double> sendBuf(pCols);
            for (int j = 0; j < pCols; ++j) {
                sendBuf[j] = xFull[p + j * size];
            }

            if (p == 0) {
                // Local copy for Rank 0
                for (int j = 0; j < pCols; ++j) xLocal[j] = sendBuf[j];
            } else {
                // Send the buffer to the target process 
                MPI_Send(sendBuf.data(), pCols, MPI_DOUBLE, p, 2, MPI_COMM_WORLD);
            }
        }
    } else if (localCols > 0) {
        // Other ranks receive their specific cyclic portion
        MPI_Recv(xLocal.get(), localCols, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return xLocal;
}

// DISTRIBUTED SpMV
void SpMVDistributed(const CSRMatrix& localCSR, const double* xLocal, const unique_ptr<double[]>& xGhost, const vector<int>& xIndex, const vector<char>& isGhost, double* y) {
    for(int i=0;i<localCSR.getRows();++i){

        double sum=0.0;
        for(int j=localCSR.getIndexPointers(i); j<localCSR.getIndexPointers(i+1); ++j){
            double val = localCSR.getData(j);

            // inline ghost check
            sum += val * (isGhost[j] ? xGhost[xIndex[j]] : xLocal[xIndex[j]]);
        }
        y[i]=sum;
    }
}

int main(int argc, char* argv[]){
    MPI_Init(&argc,&argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    ResultsManager rm;
    CLIOptions opts;

    CSRMatrix localCSR;
    vector<Entry> allEntries, localEntries;
    unique_ptr<double[]> xLocal, xGhost, yLocal;
    GhostManager gm = GhostManager(rank, size);

    int iterations=0, matrixRows=0, matrixCols=0, localCols=0;

    MPI_Datatype entryType = createEntryType();
    double globalTime=0.0, t;

    try{
        if(rank==0){
            opts = parseCLI(argc,argv);
            iterations = opts.iterations;

            allEntries = opts.generateMatrix ? generateMatrixEntries(opts.rows,opts.cols,opts.density) : readMTX(opts.filepath);

            matrixRows = matrixCols = 0;
            for(const auto& e: allEntries){
                if(e.row+1>matrixRows) matrixRows=e.row+1;
                if(e.col+1>matrixCols) matrixCols=e.col+1;
            }

            rm.setMatrixInfo(opts.filepath.empty()?"Generated":opts.filepath, opts.generateMatrix, matrixRows, matrixCols, allEntries.size(), opts.density);
            rm.setMPIInfo(size);
        }
        t = MPI_Wtime(); // setup timer

        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        distributeMatrix(allEntries, localEntries, rank, size, entryType);
        localCSR.buildFromEntries(localEntries);

        // Local NNZ stats
        size_t localNNZ = localCSR.getNNZ(), minNNZ,maxNNZ,sumNNZ;
        MPI_Reduce(&localNNZ,&minNNZ,1,MPI_UNSIGNED_LONG,MPI_MIN,0,MPI_COMM_WORLD);
        MPI_Reduce(&localNNZ,&maxNNZ,1,MPI_UNSIGNED_LONG,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&localNNZ,&sumNNZ,1,MPI_UNSIGNED_LONG,MPI_SUM,0,MPI_COMM_WORLD);
        if(rank==0) rm.setNNZStats(minNNZ, static_cast<double>(sumNNZ)/size, maxNNZ);

        int localRows = matrixRows / size + (rank < (matrixRows % size) ? 1 : 0);
        yLocal = make_unique<double[]>(max(1, localRows));

        xLocal = distributeVector(rank, size, matrixCols, localCols);

        if(size > 1) gm.setup(localCSR);   // prepare ghost indices

        gm.preIndex(localCSR);

        t = (MPI_Wtime()-t)*1e3;
        if(rank==0) rm.setSetupDuration(t);

        // Exchange ghost values if more than 1 rank
        if(size>1){
            t = MPI_Wtime();
            xGhost = gm.exchangeGhostValues(xLocal.get(), localCols);
            t = (MPI_Wtime()-t)*1e3;

            MPI_Reduce(&t,&globalTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
            if(rank==0) rm.setCommunicationDuration(globalTime);
        }

        // Perform SpMV iterations (warm-up included)
        for(int iter=-1; iter<iterations; iter++){
            t = MPI_Wtime();
            SpMVDistributed(localCSR, xLocal.get(), xGhost, gm.getXIndex(), gm.getIsGhost(), yLocal.get());
            t = (MPI_Wtime()-t)*1e3;

            MPI_Reduce(&t,&globalTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

            if(rank==0){
                if(iter==-1) rm.setWarmupDuration(globalTime);
                else rm.addKernelDuration(globalTime);
            }
        }

        if(rank==0){
            rm.computeMetrics();
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