#include "ResultsManager.h"

// -------------------- Helpers --------------------
double ResultsManager::percentile90(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::ceil(0.9 * v.size())) - 1;
    return std::min(v[idx], v.back());
}

// -------------------- Setters --------------------
void ResultsManager::setMatrixInfo(const std::string& name, bool generated,
                                   size_t r, size_t c, size_t n, double dens) {
    matrixName = name;
    isGenerated = generated;
    rows = r;
    cols = c;
    nnz = n;
    density = dens;
}

void ResultsManager::setMPIInfo(int procs) {
    mpiProcesses = procs;
}

void ResultsManager::setSetupDuration(double ms) {
    if (ms < 0.0) throw runtime_error("Setup duration cannot be negative");
    setupDuration = ms;
}

void ResultsManager::setWarmupDuration(double ms) {
    if (ms < 0.0) throw runtime_error("Warmup duration cannot be negative");
    warmupDuration = ms;
}

void ResultsManager::addKernelDuration(double ms) {
    if (ms < 0.0) throw runtime_error("Kernel duration cannot be negative");
    kernelDurations.push_back(ms);
}

void ResultsManager::setCommunicationDuration(double ms) {
    if (ms < 0.0) throw runtime_error("Communication duration cannot be negative");
    communicationDuration = ms;
}

void ResultsManager::setGhostStats(size_t minGhosts, size_t avgGhosts, size_t maxGhosts, size_t sumGhosts) {
    minGhostEntries = minGhosts;
    avgGhostEntries = avgGhosts;
    maxGhostEntries = maxGhosts;
    totalGhostEntries = sumGhosts;
}

void ResultsManager::setNNZStats(size_t minN, size_t avgN, size_t maxN) {
    minNNZ = minN;
    avgNNZ = avgN;
    maxNNZ = maxN;
}

// -------------------- Metrics computation --------------------
void ResultsManager::computeMetrics() {
    // Safety checks
    if (nnz == 0 || kernelDurations.empty() || mpiProcesses == 0) return;

    // 1. Total FLOPs: 2 per non-zero element
    totalFlops = 2 * nnz;

    // 2. Calculate bytes moved based on actual data per rank
    // CSR format (values + indices)
    size_t csrValAndInd = nnz * (sizeof(double) + sizeof(int));
    // Row pointers per rank: each rank has rows/mpiProcesses rows
    size_t rowsPerRank = (rows + mpiProcesses - 1) / mpiProcesses; 
    size_t csrRowPtr = (rowsPerRank + 1) * mpiProcesses * sizeof(int); 

    // X vector: local reads + ghost reads
    size_t xLocalBytes = nnz * sizeof(double); // estimate from local values
    size_t ghostBytes = totalGhostEntries * sizeof(double); // ghost buffer
    size_t xAccesses = xLocalBytes + ghostBytes;

    // Y vector: local writes
    size_t yWrites = rowsPerRank * mpiProcesses * sizeof(double);

    // LUT for ghost mapping
    size_t lutExtraBytes = totalGhostEntries * sizeof(int);

    totalBytesMoved = csrValAndInd + csrRowPtr + xAccesses + yWrites + lutExtraBytes;

    // 3. Average footprint per rank
    memoryFootprintBytes = totalBytesMoved / mpiProcesses;

    // 4. Kernel duration: 90th percentile
    kernelDuration90 = percentile90(kernelDurations);

    // 5. Performance metrics
    double seconds = kernelDuration90 / 1000.0;
    if (seconds > 0.0) {
        gflops = static_cast<double>(totalFlops) / seconds / 1e9;
        bandwidthGBps = static_cast<double>(totalBytesMoved) / seconds / 1e9;
    }
}


// -------------------- JSON output --------------------
string ResultsManager::toJSON() const {
    stringstream ss;
    ss << fixed << setprecision(4);

    ss << "{\n";

    ss << "  \"matrix\": {\n";
    ss << "    \"name\": \"" << matrixName << "\",\n";
    ss << "    \"rows\": " << rows << ",\n";
    ss << "    \"cols\": " << cols << ",\n";
    ss << "    \"nnz\": " << nnz << "\n";
    ss << "  },\n";

    ss << "  \"mpi\": {\n";
    ss << "    \"processes\": " << mpiProcesses << "\n";
    ss << "    \"nnz_per_rank\": {\n";
    ss << "      \"min\": " << minNNZ << ",\n";
    ss << "      \"avg\": " << avgNNZ << ",\n";
    ss << "      \"max\": " << maxNNZ << "\n";
    ss << "    },\n";
    ss << "    \"ghost_entries_per_rank\": {\n";
    ss << "      \"min\": " << minGhostEntries << ",\n";
    ss << "      \"avg\": " << avgGhostEntries << ",\n";
    ss << "      \"max\": " << maxGhostEntries << "\n";
    ss << "    },\n";
    ss << "    \"total_ghost_entries\": " << totalGhostEntries << "\n"; 
    ss << "  },\n";

    ss << "  \"timings_ms\": {\n";
    ss << "    \"setup\": " << setupDuration << ",\n";
    ss << "    \"warmup\": " << warmupDuration << ",\n";

    ss << "    \"kernel\": [";
    for (size_t i = 0; i < kernelDurations.size(); ++i) {
        ss << kernelDurations[i];
        if (i + 1 < kernelDurations.size()) ss << ", ";
    }
    ss << "],\n";

    ss << "    \"communication\": " << communicationDuration << "\n";
    ss << "  },\n";

    ss << "  \"statistics\": {\n";
    ss << "    \"kernel_p90_ms\": " << kernelDuration90 << ",\n";
    ss << "    \"gflops\": " << gflops << ",\n";
    ss << "    \"bandwidth_gbps\": " << bandwidthGBps << "\n";
    ss << "  },\n";

    ss << "  \"memory\": {\n";
    ss << "    \"bytes_per_rank\": " << memoryFootprintBytes << "\n";
    ss << "  },\n";

    ss << "  \"errors\": [";
    for (size_t i = 0; i < errors.size(); ++i) {
        ss << "\"" << errors[i] << "\"";
        if (i + 1 < errors.size()) ss << ", ";
    }
    ss << "]\n";

    ss << "}\n";
    return ss.str();
}

// -------------------- Errors & reset --------------------
void ResultsManager::addError(const std::string& msg) {
    if (msg.empty()) throw std::runtime_error("Empty error message");
    errors.push_back(msg);
}

void ResultsManager::clear() {
    kernelDurations.clear();
    communicationDuration = 0.0;
    errors.clear();
}