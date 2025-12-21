#include "ResultsManager.h"
#include <numeric>
#include <iomanip>

void ResultsManager::setMatrixInfo(const string& name, bool generated, size_t r, size_t c, size_t n, double dens) {
    matrixName = name;
    isGenerated = generated;
    rows = r;
    cols = c;
    nnz = n;
    density = dens;
}

void ResultsManager::setMPIInfo(int procs, int nNodes, double mem, const string& notes) {
    mpiProcesses = procs;
    nodes = nNodes;
    totalMemoryGB = mem;
    runNotes = notes;
}

void ResultsManager::addIterationDuration(double duration) {
    iterationDurations.push_back(duration);
}

void ResultsManager::setWarmupDuration(double duration) {
    if (duration < 0.0) throw runtime_error("Warmup duration cannot be negative.");
    warmupDuration = duration;
}

void ResultsManager::setMPIOverhead(double commTimeMs) {
    if (iterationDurations.empty()) return;
    
    // Average total time to compute percentage
    double totalTime = std::accumulate(iterationDurations.begin(), iterationDurations.end(), 0.0);
    double avgTotal = totalTime / iterationDurations.size();
    
    if (avgTotal > 0) {
        mpiOverheadPct = (commTimeMs / avgTotal) * 100.0;
    }
}

void ResultsManager::computeAllMetrics() {
    if (nnz == 0 || rows == 0) throw runtime_error("Invalid matrix dimensions for metrics.");
    if (iterationDurations.empty()) throw runtime_error("No iterations recorded.");

    // Standard SpMV FLOPs: 2 * NNZ
    totalFlops = 2 * nnz;
    
    // Standard CSR Bytes Moved estimation
    size_t bytesRead = (sizeof(double) + sizeof(int)) * nnz + sizeof(int) * (rows + 1) + sizeof(double) * cols;
    size_t bytesWritten = sizeof(double) * rows;
    totalBytesMoved = bytesRead + bytesWritten;

    // Sorting for P90
    vector<double> sortedDur = iterationDurations;
    sort(sortedDur.begin(), sortedDur.end());
    
    size_t idx90 = size_t(ceil(0.9 * sortedDur.size())) - 1;
    duration90 = sortedDur[std::min(idx90, sortedDur.size() - 1)];
    
    // Average duration
    avgDuration = std::accumulate(sortedDur.begin(), sortedDur.end(), 0.0) / sortedDur.size();

    // Performance metrics based on P90 (more conservative/stable for reports)
    double seconds = duration90 / 1000.0;
    gflops = (totalFlops / seconds) / 1e9;
    bandwidthGBps = (totalBytesMoved / seconds) / 1e9;
}

string ResultsManager::toJSON() const {
    stringstream ss;
    ss << fixed << setprecision(4);
    ss << "{\n";

    // 1. Matrix Info
    ss << "  \"matrix\": {\n"
       << "    \"name\": \"" << matrixName << "\",\n"
       << "    \"is_generated\": " << (isGenerated ? "true" : "false") << ",\n"
       << "    \"rows\": " << rows << ",\n"
       << "    \"cols\": " << cols << ",\n"
       << "    \"nnz\": " << nnz << ",\n"
       << "    \"density\": " << density << "\n"
       << "  },\n";

    // 2. Execution Environment
    ss << "  \"execution_env\": {\n"
       << "    \"mpi_processes\": " << mpiProcesses << ",\n"
       << "    \"nodes\": " << nodes << ",\n"
       << "    \"total_memory_gb\": " << totalMemoryGB << ",\n"
       << "    \"notes\": \"" << runNotes << "\"\n"
       << "  },\n";

    // 3. Benchmarking
    ss << "  \"benchmarking\": {\n"
       << "    \"warmup_time_ms\": " << warmupDuration << ",\n"
       << "    \"iterations\": " << iterationDurations.size() << ",\n"
       << "    \"all_iteration_times_ms\": [";
    for (size_t i = 0; i < iterationDurations.size(); ++i) {
        ss << iterationDurations[i] << (i == iterationDurations.size() - 1 ? "" : ", ");
    }
    ss << "]\n  },\n";

    // 4. Statistics
    ss << "  \"statistics\": {\n"
       << "    \"avg_duration_ms\": " << avgDuration << ",\n"
       << "    \"p90_duration_ms\": " << duration90 << ",\n"
       << "    \"total_flops\": " << totalFlops << ",\n"
       << "    \"total_gflops\": " << gflops << ",\n"
       << "    \"total_bandwidth_gbps\": " << bandwidthGBps << ",\n"
       << "    \"mpi_overhead_pct\": " << mpiOverheadPct << "\n"
       << "  },\n";

    // 5. Errors
    ss << "  \"errors\": [";
    for (size_t i = 0; i < errors.size(); ++i) {
        ss << "\"" << errors[i] << "\"" << (i == errors.size() - 1 ? "" : ", ");
    }
    ss << "]\n";

    ss << "}";
    return ss.str();
}

void ResultsManager::addError(const string& msg) {
    if (!msg.empty()) errors.push_back(msg);
}

void ResultsManager::clear() {
    iterationDurations.clear();
    errors.clear();
    warmupDuration = 0.0;
    duration90 = 0.0;
    mpiOverheadPct = 0.0;
}