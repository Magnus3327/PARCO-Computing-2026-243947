#include "GhostManager.h"
#include <numeric>
#include <algorithm>

using namespace std;

void GhostManager::setup(const CSRMatrix& localCSR) {
    sendCounts.assign(size, 0);
    recvCounts.assign(size, 0);
    neededCols.assign(size, {});
    recvCols.assign(size, {});

    // Identify needed ghost columns
    for (int i = 0; i < localCSR.getRows(); ++i) {
        for (int j = localCSR.getIndexPointers(i); j < localCSR.getIndexPointers(i + 1); ++j) {

            int col   = localCSR.getIndices(j);
            int owner = col % size;

            if (owner != rank) neededCols[owner].push_back(col);
        }
    }

    // Remove duplicates and compute send counts
    for (int p = 0; p < size; ++p) {
        auto& v = neededCols[p];
        sort(v.begin(), v.end());
        v.erase(unique(v.begin(), v.end()), v.end());
        sendCounts[p] = static_cast<int>(v.size());
    }

    // Exchange counts
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Exchange column indices
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

    if (!reqs.empty())
        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Build ghost offsets (FLAT layout)
    ghostOffsets.resize(size + 1, 0);
    for (int p = 0; p < size; ++p) ghostOffsets[p + 1] = ghostOffsets[p] + sendCounts[p];

    totalGhostCount = ghostOffsets[size];
}
// Pre-index: To velocize spmv kernel, without losing time into hashing
void GhostManager::preIndex(const CSRMatrix& localCSR) {
    const size_t nnz = localCSR.getNNZ();
    xIndex.resize(nnz);
    isGhost.resize(nnz);

    if (size == 1) {
        for (size_t j = 0; j < nnz; ++j) {
            xIndex[j] = localCSR.getIndices(j);
            isGhost[j] = 0;
        }
        return;
    }

    // Build implicit ghost mapping via offsets
    vector<int> ghostPos(size, 0);

    for (int p = 0; p < size; ++p) ghostPos[p] = ghostOffsets[p];

    for (int i = 0; i < localCSR.getRows(); ++i) {
        for (int j = localCSR.getIndexPointers(i);j < localCSR.getIndexPointers(i + 1); ++j) {
            int col   = localCSR.getIndices(j);
            int owner = col % size;

            if (owner == rank) {
                xIndex[j] = (col - rank) / size;
                isGhost[j] = 0;
            } else {
                int idx = ghostPos[owner]++;
                xIndex[j] = idx;
                isGhost[j] = 1;
            }
        }
    }
}

// Exchange: Using non blocking send/receive
unique_ptr<double[]> GhostManager::exchangeGhostValues(const double* xLocal, int localCols) {
    if (size == 1) return nullptr;

    vector<MPI_Request> reqs;

    vector<double> sendBufFlat;
    vector<double> recvBufFlat(totalGhostCount);

    // Pack send buffer
    int sendTotal = accumulate(recvCounts.begin(), recvCounts.end(), 0);
    sendBufFlat.resize(sendTotal);

    vector<int> sendOffsets(size, 0);
    for (int p = 1; p < size; ++p)
        sendOffsets[p] = sendOffsets[p - 1] + recvCounts[p - 1];

    for (int p = 0; p < size; ++p) {
        for (int i = 0; i < recvCounts[p]; ++i) {
            int col = recvCols[p][i];
            sendBufFlat[sendOffsets[p] + i] = xLocal[(col - rank) / size];
        }
    }

    // Post receives
    for (int p = 0; p < size; ++p) {
        if (sendCounts[p] > 0) {
            MPI_Request r;
            MPI_Irecv(recvBufFlat.data() + ghostOffsets[p], sendCounts[p], MPI_DOUBLE, p, 4, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }

    // Post sends
    for (int p = 0; p < size; ++p) {
        if (recvCounts[p] > 0) {
            MPI_Request r;
            MPI_Isend(sendBufFlat.data() + sendOffsets[p], recvCounts[p], MPI_DOUBLE, p, 4, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }

    if (!reqs.empty())
        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    unique_ptr<double[]> xGhost(new double[max(1, totalGhostCount)]);
    copy(recvBufFlat.begin(), recvBufFlat.end(), xGhost.get());

    return xGhost;
}