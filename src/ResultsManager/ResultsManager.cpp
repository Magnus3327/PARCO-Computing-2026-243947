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

void ResultsManager::setMPIInfo(int procs) {
    mpiProcesses = procs;
}

void ResultsManager::addIterationDuration(double duration) {
    iterationDurations.push_back(duration);
}

void ResultsManager::computeAllMetrics() {
    if (nnz == 0 || iterationDurations.empty()) return;

    // SpMV esegue 1 Moltiplicazione e 1 Somma per ogni NNZ
    totalFlops = 2 * nnz; 
    
    // Stima traffico dati (Valori + Indici + RowPtr + Vettori x e y)
    size_t bytesRead = (sizeof(double) + sizeof(int)) * nnz + sizeof(int) * (rows + 1) + sizeof(double) * cols;
    size_t bytesWritten = sizeof(double) * rows;
    totalBytesMoved = bytesRead + bytesWritten;

    vector<double> sortedDur = iterationDurations;
    sort(sortedDur.begin(), sortedDur.end());
    
    avgDuration = std::accumulate(sortedDur.begin(), sortedDur.end(), 0.0) / sortedDur.size();
    
    size_t idx90 = size_t(ceil(0.9 * sortedDur.size())) - 1;
    duration90 = sortedDur[std::min(idx90, sortedDur.size() - 1)];

    // Performance basata sulla durata media (ms -> s)
    double seconds = avgDuration / 1000.0;
    if (seconds > 0) {
        gflops = (totalFlops / seconds) / 1e9;
        bandwidthGBps = (totalBytesMoved / seconds) / 1e9;
    }
}

string ResultsManager::toJSON() const {
    stringstream ss;
    ss << fixed << setprecision(4);
    ss << "{\n  \"matrix\": {\n"
       << "    \"name\": \"" << matrixName << "\",\n"
       << "    \"rows\": " << rows << ",\n"
       << "    \"cols\": " << cols << ",\n"
       << "    \"nnz\": " << nnz << "\n  },\n";

    ss << "  \"execution_env\": {\n"
       << "    \"mpi_processes\": " << mpiProcesses << "\n  },\n";

    ss << "  \"statistics\": {\n"
       << "    \"avg_duration_ms\": " << avgDuration << ",\n"
       << "    \"p90_duration_ms\": " << duration90 << ",\n"
       << "    \"total_gflops\": " << gflops << ",\n"
       << "    \"bandwidth_gbps\": " << bandwidthGBps << "\n  },\n";

    ss << "  \"errors\": [";
    for (size_t i = 0; i < errors.size(); ++i) {
        ss << "\"" << errors[i] << "\"" << (i == errors.size() - 1 ? "" : ", ");
    }
    ss << "]\n}";
    return ss.str();
}

void ResultsManager::addError(const string& msg) {
    errors.push_back(msg);
}

void ResultsManager::clear() {
    iterationDurations.clear();
    errors.clear();
}