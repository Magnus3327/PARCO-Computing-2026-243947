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

// Project headers (UNITN HPC Project)
#include "ResultsManager/ResultsManager.h"
#include "MTX/MTXReader.h"
#include "CSR/CSRMatrix.h"
#include "Utils/Utils.h"

using namespace std;
using namespace mtx;
using namespace utils;

/**
 * Struttura per gestire i dati dei Ghost.
 * Permette di separare la fase di analisi (search) da quella di scambio (exchange).
 */
struct GhostData {
    vector<vector<int>> neededCols; 
    vector<int> sendCounts;         
    vector<int> recvCounts;         
    unique_ptr<double[]> xGhost;
    unordered_map<int, int> ghostMap;       
};

// MPI Datatype per le Entry del formato COO
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

// Distribuzione 1D Cyclic della matrice
void distributeMatrix(const vector<Entry>& allEntries, vector<Entry>& localEntries, int rank, int size, MPI_Datatype entryType) {
    if(rank==0) {
        vector<vector<Entry>> buckets(size);
        for(const auto& e: allEntries){
            int owner = e.row % size;
            Entry local = e;
            local.row = (e.row - owner)/size;
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
 * Fase di Scoperta (Search): identifica quali colonne servono e da chi.
 * Eseguita una sola volta prima del benchmark.
 */
void searchGhostValues(int rank, int size, const CSRMatrix& localCSR, GhostData& gd) {
    gd.neededCols.assign(size, vector<int>());
    gd.sendCounts.assign(size, 0);
    gd.recvCounts.assign(size, 0);

    for(int i=0; i<localCSR.getRows(); ++i){
        for(int j=localCSR.getIndexPointers(i); j<localCSR.getIndexPointers(i+1); ++j){
            int col = localCSR.getIndices(j);
            int owner = col % size;
            if(owner != rank) gd.neededCols[owner].push_back(col);
        }
    }

    for(int p=0; p<size; ++p){
        auto& v = gd.neededCols[p];
        sort(v.begin(), v.end());
        v.erase(unique(v.begin(), v.end()), v.end());
        gd.sendCounts[p] = static_cast<int>(v.size());
    }
    MPI_Alltoall(gd.sendCounts.data(), 1, MPI_INT, gd.recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
}

/**
 * Fase di Scambio (Exchange): scambia effettivamente i valori double di x.
 */
void exchangeGhostValues(int rank, int size, GhostData& gd, const double* xLocal, int localCols) {
    if (size <= 1) return;
    vector<vector<int>> recvCols(size);
    vector<MPI_Request> reqs;
    reqs.reserve(size * 2);

    for (int p = 0; p < size; ++p) {
        if (gd.recvCounts[p] > 0) {
            recvCols[p].resize(gd.recvCounts[p]);
            MPI_Request r;
            MPI_Irecv(recvCols[p].data(), gd.recvCounts[p], MPI_INT, p, 3, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
        if (gd.sendCounts[p] > 0) {
            MPI_Request r;
            MPI_Isend(gd.neededCols[p].data(), gd.sendCounts[p], MPI_INT, p, 3, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }
    if (!reqs.empty()) {
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        reqs.clear();
    }

    vector<vector<double>> sendVals(size);
    for (int p = 0; p < size; ++p) {
        if (gd.recvCounts[p] == 0) continue;
        sendVals[p].resize(gd.recvCounts[p]);
        for (int i = 0; i < gd.recvCounts[p]; ++i) {
            sendVals[p][i] = xLocal[(recvCols[p][i] - rank) / size];
        }
    }

    vector<vector<double>> recvVals(size);
    for (int p = 0; p < size; ++p) {
        if (gd.sendCounts[p] > 0) {
            recvVals[p].resize(gd.sendCounts[p]);
            MPI_Request r;
            MPI_Irecv(recvVals[p].data(), gd.sendCounts[p], MPI_DOUBLE, p, 4, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
        if (gd.recvCounts[p] > 0) {
            MPI_Request r;
            MPI_Isend(sendVals[p].data(), gd.recvCounts[p], MPI_DOUBLE, p, 4, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }
    if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    int totalGhosts = accumulate(gd.sendCounts.begin(), gd.sendCounts.end(), 0);
    gd.xGhost = make_unique<double[]>(max(1, totalGhosts));
    gd.ghostMap.clear();
    int pos = 0;
    for (int p = 0; p < size; ++p) {
        for (size_t i = 0; i < recvVals[p].size(); ++i) {
            gd.ghostMap[gd.neededCols[p][i]] = pos;
            gd.xGhost[pos++] = recvVals[p][i];
        }
    }
}

/**
 * Pre-indexing: mappa gli indici globali in indici locali o ghost.
 * Elimina la necessità di ricerche in mappe hash durante il loop SpMV.
 */
void preIndexVector(const CSRMatrix& localCSR, int rank, int size, const GhostData& gd, vector<int>& xIndex, vector<char>& isGhost, int localCols) {
    const size_t nnz = localCSR.getNNZ();
    xIndex.resize(nnz);
    isGhost.resize(nnz);

    if (size <= 1) {
        for (size_t j = 0; j < nnz; ++j) {
            isGhost[j] = 0;
            xIndex[j] = localCSR.getIndices(j);
        }
        return;
    }

    for(int i=0; i<localCSR.getRows(); ++i){
        for(int j=localCSR.getIndexPointers(i); j<localCSR.getIndexPointers(i+1); ++j){
            int col = localCSR.getIndices(j);
            if(col % size == rank){
                isGhost[j] = 0;
                xIndex[j] = (col - rank) / size;
            } else {
                isGhost[j] = 1;
                xIndex[j] = gd.ghostMap.at(col);
            }
        }
    }
}

void SpMVDistributed(const CSRMatrix& localCSR, const double* xLocal, const double* xGhost, const vector<int>& xIndex, const vector<char>& isGhost, double* y) {
    for(int i=0; i<localCSR.getRows(); ++i){
        double sum = 0.0;
        for(int j=localCSR.getIndexPointers(i); j<localCSR.getIndexPointers(i+1); ++j){
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
    GhostData gd;

    int iterations=0, matrixRows=0, matrixCols=0, localCols=0;
    MPI_Datatype entryType = createEntryType();
    double globalTime=0.0, time;

    try{
        if(rank==0){
            opts = parseCLI(argc, argv);
            iterations = opts.iterations;
            allEntries = opts.generateMatrix ? generateMatrixEntries(opts.rows, opts.cols, opts.density) : readMTX(opts.filepath);
            matrixRows = 0; matrixCols = 0;
            for(const auto& e: allEntries) {
                if(e.row+1 > matrixRows) matrixRows = e.row+1;
                if(e.col+1 > matrixCols) matrixCols = e.col+1;
            }
            rm.setMatrixInfo(opts.filepath.empty()?"Generated":opts.filepath, opts.generateMatrix, matrixRows, matrixCols, allEntries.size(), opts.density);
            rm.setMPIInfo(size);
        }

        time = MPI_Wtime();
        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        distributeMatrix(allEntries, localEntries, rank, size, entryType);
        localCSR.buildFromEntries(localEntries);
        x = distributeVector(rank, size, matrixCols, localCols);

        size_t localNNZ = localCSR.getNNZ(); 
        size_t minNNZ, maxNNZ, sumNNZ;

        MPI_Reduce(&localNNZ, &minNNZ, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localNNZ, &maxNNZ, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localNNZ, &sumNNZ, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank == 0) rm.setNNZStats(minNNZ, static_cast<double>(sumNNZ)/size, maxNNZ);

        // Se size > 1 facciamo la ricerca, altrimenti saltiamo tutto
        if(size > 1) searchGhostValues(rank, size, localCSR, gd);

        time = (MPI_Wtime() - time) * 1e3;
        if(rank==0) rm.setSetupDuration(time);

        time = MPI_Wtime();
        if(size > 1) exchangeGhostValues(rank, size, gd, x.get(), localCols);
        time = (MPI_Wtime() - time) * 1e3;
        MPI_Reduce(&time, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(rank==0) rm.setCommunicationDuration(globalTime);

        time = MPI_Wtime();
        preIndexVector(localCSR, rank, size, gd, xIndex, isGhost, localCols);
        time = (MPI_Wtime() - time) * 1e3;
        MPI_Reduce(&time, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(rank==0) rm.addSetupDuration(globalTime);
        
        size_t localGhosts = (size > 1) ? gd.ghostMap.size() : 0;
        size_t minG, maxG, sumG;
        MPI_Reduce(&localGhosts, &minG, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localGhosts, &maxG, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localGhosts, &sumG, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if(rank==0) rm.setGhostStats(minG, (size > 1 ? (double)sumG/size : 0), maxG, sumG);

        int localRows = matrixRows / size + (rank < (matrixRows % size) ? 1 : 0);
        yLocal = make_unique<double[]>(max(1, localRows));

        // SpMV Kernel Loop
        for(int iter=-1; iter<iterations; iter++){
            time = MPI_Wtime();
            SpMVDistributed(localCSR, x.get(), gd.xGhost.get(), xIndex, isGhost, yLocal.get());
            time = (MPI_Wtime()-time)*1e3;
            MPI_Reduce(&time, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if(iter>=0 && rank==0) rm.addKernelDuration(globalTime);
            else if(iter==-1 && rank==0) rm.setWarmupDuration(globalTime);
        }

        if(rank==0) { rm.computeMetrics(); cout << rm.toJSON() << endl; }
    } catch(const exception& e){
        if(rank==0){ rm.addError(e.what()); cout << rm.toJSON() << endl; }
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Type_free(&entryType);
    MPI_Finalize();
    return 0;
}