#include "GhostManager.h"

// Setup: Identify the ghost values
void GhostManager::setup(const CSRMatrix& localCSR) {
    sendCounts.resize(size, 0);
    recvCounts.resize(size, 0);
    neededCols.resize(size);

    // Identify remote columns for each rank
    for (int i = 0; i < localCSR.getRows(); ++i)
        for (int j = localCSR.getIndexPointers(i); j < localCSR.getIndexPointers(i+1); ++j) {
            int col = localCSR.getIndices(j);
            int owner = col % size;
            if (owner != rank) neededCols[owner].push_back(col);
        }

    // Remove duplicates and compute send counts
    for (int p = 0; p < size; ++p) {
        sort(neededCols[p].begin(), neededCols[p].end());
        neededCols[p].erase(unique(neededCols[p].begin(), neededCols[p].end()), neededCols[p].end());
        sendCounts[p] = static_cast<int>(neededCols[p].size());
    }

    // Exchange counts with all ranks
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Exchange ghost indices with non-blocking communication
    recvCols.resize(size);
    vector<MPI_Request> reqs;
    for (int p = 0; p < size; ++p) {
        if (recvCounts[p] > 0) {
            recvCols[p].resize(recvCounts[p]);
            MPI_Request r;
            MPI_Irecv(recvCols[p].data(), recvCounts[p], MPI_INT, p, 3, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
        if (sendCounts[p] > 0) {
            MPI_Request r;
            MPI_Isend(neededCols[p].data(), sendCounts[p], MPI_INT, p, 3, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }
    if (!reqs.empty()) MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Total number of ghost elements
    totalGhostCount = accumulate(sendCounts.begin(), sendCounts.end(), 0);
}

// Pre-index: To velocize spmv kernel, without losing time into hashing
void GhostManager::preIndex(const CSRMatrix& localCSR) {
    const size_t nnz = localCSR.getNNZ();
    xIndex.resize(nnz);
    isGhost.resize(nnz);

    // If only one rank, everything is local
    if (size == 1) {
        for (int i = 0; i < localCSR.getRows(); ++i)
            for (int j = localCSR.getIndexPointers(i); j < localCSR.getIndexPointers(i+1); ++j)
                xIndex[j] = localCSR.getIndices(j), isGhost[j] = 0;
        return;
    }

    // Map ghost columns to local indices
    unordered_map<int,int> ghostMap;
    int pos = 0;
    for (int p = 0; p < size; ++p)
        for (int col : neededCols[p])
            ghostMap[col] = pos++;

    // Pre-index local and ghost entries
    for (int i = 0; i < localCSR.getRows(); ++i)
        for (int j = localCSR.getIndexPointers(i); j < localCSR.getIndexPointers(i+1); ++j) {
            int col = localCSR.getIndices(j);
            if (col % size == rank) {
                xIndex[j] = (col - rank) / size;
                isGhost[j] = 0;
            } else {
                auto it = ghostMap.find(col);
                if (it == ghostMap.end()) throw runtime_error("Ghost column missing in plan");
                xIndex[j] = it->second;
                isGhost[j] = 1;
            }
        }
}

// Exchange: Using non blocking send/receive
unique_ptr<double[]> GhostManager::exchangeGhostValues(const double* xLocal, int localCols) {
    if (size == 1) return nullptr;

    vector<MPI_Request> reqs;
    vector<vector<double>> sendBuf(size), recvBuf(size);

    // Pack data to send
    for (int p = 0; p < size; ++p) {
        if (recvCounts[p] == 0) continue;
        sendBuf[p].resize(recvCounts[p]);
        for (int i = 0; i < recvCounts[p]; ++i)
            sendBuf[p][i] = xLocal[(recvCols[p][i] - rank) / size];
    }

    // Non-blocking communication
    for (int p = 0; p < size; ++p) {
        if (sendCounts[p] > 0) {
            recvBuf[p].resize(sendCounts[p]);
            MPI_Request r;
            MPI_Irecv(recvBuf[p].data(), sendCounts[p], MPI_DOUBLE, p, 4, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
        if (recvCounts[p] > 0) {
            MPI_Request r;
            MPI_Isend(sendBuf[p].data(), recvCounts[p], MPI_DOUBLE, p, 4, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }
    if (!reqs.empty()) MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Flatten received ghosts into a contiguous buffer
    unique_ptr<double[]> xGhost(new double[max(1, totalGhostCount)]);
    int pos = 0;
    for (int p = 0; p < size; ++p)
        for (size_t i = 0; i < recvBuf[p].size(); ++i)
            xGhost[pos++] = recvBuf[p][i];

    return xGhost;
}