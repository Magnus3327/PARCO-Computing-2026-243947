#include "ResultsManager.h"
#include <numeric>
#include <iomanip>

// Set global matrix info (Rank 0 only)
void ResultsManager::setMatrixInfo(const string& name, bool generated, size_t r, size_t c, size_t n, double dens) {
    matrixName = name;
    isGenerated = generated;
    rows = r;
    cols = c;
    nnz = n;
    density = dens;
}

// Set MPI environment info
void ResultsManager::setMPIInfo(int procs, int nNodes, double mem, const string& notes) {
    mpiProcesses = procs;
    nodes = nNodes;
    totalMemoryGB = mem;
    runNotes = notes;
}

void ResultsManager::addIterationDuration(double duration) {
    iterationDurations.push_back(duration); //
}

void ResultsManager::setMPIOverhead(double commTimeMs) {
    if (iterationDurations.empty()) return;
    double avgTotal = std::accumulate(iterationDurations.begin(), iterationDurations.end(), 0.0) / iterationDurations.size();
    if (avgTotal > 0) mpiOverheadPct = (commTimeMs / avgTotal) * 100.0;
}

void ResultsManager::computeAllMetrics() {
    if (nnz == 0 || rows == 0) throw runtime_error("Invalid matrix dimensions for metrics."); //
    
    totalFlops = 2 * nnz; // Standard SpMV: multiply and add for each NNZ
    
    // Estimate bytes based on global CSR structure
    size_t bytesRead = (sizeof(double) + sizeof(int)) * nnz + sizeof(int) * (rows + 1) + sizeof(double) * cols;
    size_t bytesWritten = sizeof(double) * rows;
    totalBytesMoved = bytesRead + bytesWritten;

    vector<double> sortedDur = iterationDurations;
    sort(sortedDur.begin(), sortedDur.end()); //
    
    size_t idx90 = size_t(ceil(0.9 * sortedDur.size())) - 1;
    duration90 = sortedDur[std::min(idx90, sortedDur.size() - 1)]; //
    avgDuration = std::accumulate(sortedDur.begin(), sortedDur.end(), 0.0) / sortedDur.size();

    double seconds = duration90 / 1000.0;
    gflops = (totalFlops / seconds) / 1e9; //
    bandwidthGBps = (totalBytesMoved / seconds) / 1e9; //
}

string ResultsManager::toJSON() const {
    stringstream ss;
    ss << fixed << setprecision(4);
    ss << "{\n  \"matrix\": {\n"
       << "    \"name\": \"" << matrixName << "\",\n"
       << "    \"is_generated\": " << (isGenerated ? "true" : "false") << ",\n"
       << "    \"rows\": " << rows << ",\n"
       << "    \"cols\": " << cols << ",\n"
       << "    \"nnz\": " << nnz << ",\n"
       << "    \"density\": " << density << "\n  },\n";

    ss << "  \"execution_env\": {\n"
       << "    \"mpi_processes\": " << mpiProcesses << ",\n"
       << "    \"nodes\": " << nodes << ",\n"
       << "    \"total_memory_gb\": " << totalMemoryGB << ",\n"
       << "    \"notes\": \"" << runNotes << "\"\n  },\n";

    ss << "  \"statistics\": {\n"
       << "    \"avg_duration_ms\": " << avgDuration << ",\n"
       << "    \"p90_duration_ms\": " << duration90 << ",\n"
       << "    \"total_gflops\": " << gflops << ",\n"
       << "    \"mpi_overhead_pct\": " << mpiOverheadPct << "\n  },\n";

    ss << "  \"errors\": [";
    for (size_t i = 0; i < errors.size(); ++i) {
        ss << "\"" << errors[i] << "\"" << (i == errors.size() - 1 ? "" : ", ");
    }
    ss << "]\n}";
    return ss.str(); //
}