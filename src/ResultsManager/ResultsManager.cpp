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

    // 1. FLOPs: 1 Moltiplicazione + 1 Somma per ogni elemento non nullo (NNZ)
    totalFlops = static_cast<size_t>(2) * nnz; 
    
    // 2. Calcolo analitico dei Bytes Mossi (Modello Standard SpMV)
    
    // Matrice CSR (Read): Valori (double) + Indici di colonna (int) + Row Pointers (int)
    size_t csrBytes = nnz * (sizeof(double) + sizeof(int)) + (rows + 1) * sizeof(int);
    
    // Vettore X (Read): In una matrice sparsa generica, ogni NNZ genera un accesso a X.
    // Anche se X è bcastato e sta in cache, lo standard di benchmarking SpMV 
    // prevede di contare 1 caricamento di double per ogni NNZ.
    size_t vectorXBytes = nnz * sizeof(double);
    
    // Vettore Y (Write): Ogni riga prodotta viene scritta una volta.
    size_t vectorYBytes = rows * sizeof(double);
    
    totalBytesMoved = csrBytes + vectorXBytes + vectorYBytes;

    // 3. Statistiche Temporali
    vector<double> sortedDur = iterationDurations;
    sort(sortedDur.begin(), sortedDur.end());
    
    avgDuration = std::accumulate(sortedDur.begin(), sortedDur.end(), 0.0) / sortedDur.size();
    
    // Percentile 90 per identificare la latenza "worst-case" stabile
    size_t idx90 = size_t(ceil(0.9 * sortedDur.size())) - 1;
    duration90 = sortedDur[std::min(idx90, sortedDur.size() - 1)];

    // 4. Calcolo GFLOPS e Bandwidth (Conversion: ms -> s, Bytes -> GB)
    double seconds = avgDuration / 1000.0;
    if (seconds > 0) {
        gflops = (static_cast<double>(totalFlops) / seconds) / 1e9;
        bandwidthGBps = (static_cast<double>(totalBytesMoved) / seconds) / 1e9;
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

    ss << "  \"benchmarking\": {\n"
       << "    \"iterations\": " << iterationDurations.size() << ",\n"
       << "    \"all_iteration_times_ms\": [";
    
    // Aggiunta dell'array con tutte le misurazioni
    for (size_t i = 0; i < iterationDurations.size(); ++i) {
        ss << iterationDurations[i] << (i == iterationDurations.size() - 1 ? "" : ", ");
    }
    
    ss << "]\n  },\n";

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