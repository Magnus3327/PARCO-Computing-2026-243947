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

// -------------------- Metrics computation --------------------
void ResultsManager::computeMetrics() {
    if (nnz == 0 || kernelDurations.empty()) return;

    // FLOPs: 2 per ogni elemento non nullo
    totalFlops = static_cast<size_t>(2) * nnz;

    // Calcolo dei Bytes Mossi
    size_t csrBytes = nnz * (sizeof(double) + sizeof(int)) + (rows + 1) * sizeof(int);
    size_t vectorXBytes = nnz * sizeof(double);
    size_t vectorYBytes = rows * sizeof(double);
    
    // Traffico LUT: per ogni ghost entry aggregata, leggiamo un indice int
    size_t lutExtraBytes = totalGhostEntries * sizeof(int);

    totalBytesMoved = csrBytes + vectorXBytes + vectorYBytes + lutExtraBytes;
    
    // Memory footprint medio per rank (quello che vedevi a 0)
    memoryFootprintBytes = totalBytesMoved / mpiProcesses;

    kernelDuration90 = percentile90(kernelDurations);

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