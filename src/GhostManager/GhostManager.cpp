#include "GhostManager.h"

/**
 * Constructor: Initializes rank and size for MPI environment.
 */
GhostManager::GhostManager(int r, int s) : rank(r), size(s) {}

/**
 * Search Phase (Discovery):
 * Identifies which non-local columns are required for local SpMV computation.
 * Performs a collective MPI_Alltoall to synchronize send/receive counts between ranks.
 */
void GhostManager::search(const CSRMatrix& localCSR) {
    if (size <= 1) return;

    // Reset and initialize structures for the current matrix
    neededCols.assign(size, vector<int>());
    sendCounts.assign(size, 0);
    recvCounts.assign(size, 0);

    // Identify needed off-process columns based on 1D cyclic distribution
    for(int i = 0; i < localCSR.getRows(); ++i){
        for(int j = localCSR.getIndexPointers(i); j < localCSR.getIndexPointers(i+1); ++j){
            int col = localCSR.getIndices(j);
            int owner = col % size;
            if(owner != rank) neededCols[owner].push_back(col);
        }
    }

    // Sort and remove duplicates to minimize communication volume
    for(int p = 0; p < size; ++p){
        sort(neededCols[p].begin(), neededCols[p].end());
        neededCols[p].erase(unique(neededCols[p].begin(), neededCols[p].end()), neededCols[p].end());
        sendCounts[p] = static_cast<int>(neededCols[p].size());
    }

    // Exchange counts with all other processes
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
}

/**
 * Exchange Phase:
 * 1. Exchanges column indices to let neighbors know which local values to send.
 * 2. Packs local x values into buffers.
 * 3. Performs non-blocking communication of actual double values.
 * 4. Maps received values into the xGhost buffer.
 */
void GhostManager::exchange(const double* xLocal, int localCols) {
    if (size <= 1) return;

    vector<vector<int>> recvCols(size);
    vector<MPI_Request> reqs;
    
    // Calculate exact number of requests to avoid vector reallocations (ensures pointer stability)
    int numReqs = 0;
    for(int p=0; p<size; ++p) {
        if(recvCounts[p] > 0) numReqs++;
        if(sendCounts[p] > 0) numReqs++;
    }
    reqs.resize(numReqs);
    int rIdx = 0;

    // --- PHASE 1: Index Exchange (Tag 3) ---
    for (int p = 0; p < size; ++p) {
        if (recvCounts[p] > 0) {
            recvCols[p].resize(recvCounts[p]);
            MPI_Irecv(recvCols[p].data(), recvCounts[p], MPI_INT, p, 3, MPI_COMM_WORLD, &reqs[rIdx++]);
        }
        if (sendCounts[p] > 0) {
            MPI_Isend(neededCols[p].data(), sendCounts[p], MPI_INT, p, 3, MPI_COMM_WORLD, &reqs[rIdx++]);
        }
    }
    if (rIdx > 0) MPI_Waitall(rIdx, reqs.data(), MPI_STATUSES_IGNORE);

    // --- PHASE 2: Local Packing ---
    // Extract local x elements that were requested by other ranks
    vector<vector<double>> sendVals(size);
    for (int p = 0; p < size; ++p) {
        if (recvCounts[p] == 0) continue;
        sendVals[p].resize(recvCounts[p]);
        for (int i = 0; i < recvCounts[p]; ++i) {
            // Translate global column index to local buffer index
            sendVals[p][i] = xLocal[(recvCols[p][i] - rank) / size];
        }
    }

    // --- PHASE 3: Double Value Exchange (Tag 4) ---
    vector<vector<double>> recvVals(size);
    rIdx = 0; // Reset request index for the next communication round

    for (int p = 0; p < size; ++p) {
        if (sendCounts[p] > 0) {
            recvVals[p].resize(sendCounts[p]);
            MPI_Irecv(recvVals[p].data(), sendCounts[p], MPI_DOUBLE, p, 4, MPI_COMM_WORLD, &reqs[rIdx++]);
        }
        if (recvCounts[p] > 0) {
            MPI_Isend(sendVals[p].data(), recvCounts[p], MPI_DOUBLE, p, 4, MPI_COMM_WORLD, &reqs[rIdx++]);
        }
    }
    if (rIdx > 0) MPI_Waitall(rIdx, reqs.data(), MPI_STATUSES_IGNORE);

    // --- PHASE 4: Ghost Buffer Assembly and Mapping ---
    int totalGhosts = accumulate(sendCounts.begin(), sendCounts.end(), 0);
    xGhost = make_unique<double[]>(max(1, totalGhosts));
    ghostMap.clear();
    
    int pos = 0;
    for (int p = 0; p < size; ++p) {
        for (size_t i = 0; i < recvVals[p].size(); ++i) {
            int globalCol = neededCols[p][i];
            ghostMap[globalCol] = pos;
            xGhost[pos++] = recvVals[p][i];
        }
    }
}

/**
 * Pre-Indexing Phase with LUT Optimization:
 * Translates global column indices into local or ghost buffer offsets.
 * Using a map lookup during this phase is acceptable, but once the indices are
 * stored in xIndex, the kernel SpMVDistributed achieves O(1) direct access.
 */
void GhostManager::buildPreIndex(const CSRMatrix& localCSR, vector<int>& xIndex, vector<char>& isGhost, int localCols) {
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

    // Distributed case mapping
    for(int i = 0; i < localCSR.getRows(); ++i){
        for(int j = localCSR.getIndexPointers(i); j < localCSR.getIndexPointers(i + 1); ++j){
            int col = localCSR.getIndices(j);
            
            if(col % size == rank){
                // Column owned by current rank (Local)
                isGhost[j] = 0;
                xIndex[j] = (col - rank) / size;
            } else {
                // Column belongs to another rank (Ghost)
                isGhost[j] = 1;
                // O(1) average lookup to find the offset in xGhost
                xIndex[j] = ghostMap.at(col);
            }
        }
    }
}