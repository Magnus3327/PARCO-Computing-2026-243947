#include "ResultsManager.h"

ResultsManager::~ResultsManager() {
    clear();
}

void ResultsManager::setInformation(CSRMatrix* csr, string mtxName) {
    csrMatrix = csr;
    matrixName = mtxName;
}

void ResultsManager::setInformation(CSRMatrix* csr, int nThreads, const string schedType, int cSize, string mtxName) {
    csrMatrix = csr;
    numThreads = nThreads;
    schedulingType = schedType;
    chunkSize = cSize;
    matrixName = mtxName;

    sequential = false;
}

void ResultsManager::addIterationDuration(double duration) {
    iterationDurations.push_back(duration);
}

void ResultsManager::setIterationDurations(const vector<double>& durations) {
    if (durations.empty())
        throw runtime_error("Iteration durations vector is empty.");
    iterationDurations = durations;
}

void ResultsManager::setWarmupDuration(double duration) {
    if (duration < 0.0)
        throw runtime_error("Warmup duration cannot be negative.");
    warmupDuration = duration;
}

void ResultsManager::setRealTimeMetrics(size_t byteMoved, size_t flopsMoved) {
    bytesMoved = byteMoved;
    flops = flopsMoved;
}

void ResultsManager::computeAllMetrics() {
    if (!csrMatrix || csrMatrix->getRows() == 0 || csrMatrix->getCols() == 0) {
        throw runtime_error("CSR matrix is empty or invalid");
    }

    size_t nnz = csrMatrix->getNNZ();
    size_t rows = csrMatrix->getRows();
    size_t cols = csrMatrix->getCols();

    // Estimate FLOPs and Bytes moved if not collected during execution
    if (flops == 0 || bytesMoved == 0) {
        flops = 2.0 * nnz; 
        size_t bytesRead = 8 * nnz           // CSR data (double)
                        + 4 * nnz            // CSR indices (int)
                        + 4 * (rows + 1)     // CSR indexPointers (int)
                        + 8 * cols;          // input vector (double)
        size_t bytesWritten = 8 * rows;      // output vector (double)
        bytesMoved = bytesRead + bytesWritten; 
    } 

    if (iterationDurations.empty())
        throw runtime_error("No iteration durations recorded. Cannot compute 90th percentile.");

    vector<double> sortedDur = iterationDurations;
    sort(sortedDur.begin(), sortedDur.end());

    size_t idx90 = size_t(ceil(0.9 * sortedDur.size())) - 1;
    if (idx90 >= sortedDur.size()) idx90 = sortedDur.size() - 1;

    duration90 = sortedDur[idx90];                  // ms
    double seconds = duration90 / 1000.0;
    
    // Performance (GFLOPS), bandwidth (GB/s), e Arithmetic Intensity (FLOP/byte)
    gflops = flops / seconds / 1e9;
    bandwidthGBps = double(bytesMoved) / (seconds * 1e9);
    arithmeticIntensity = flops / double(bytesMoved);
}

void ResultsManager::addError(const string& msg) {
    if (msg.empty())
        throw runtime_error("Error message cannot be empty.");
    errors.push_back(msg);
}

string ResultsManager::toJSON() const {
    stringstream ss;
    ss << "{\n";

    // Matrix info
    ss << "  \"matrix\": {\n";
    if (csrMatrix) { // if the reading of the file failed, then in main->catch we print an empty matrix info and the error
        ss << "    \"name\": \"" << matrixName << "\",\n";
        ss << "    \"rows\": " << csrMatrix->getRows() << ",\n";
        ss << "    \"cols\": " << csrMatrix->getCols() << ",\n";
        ss << "    \"nnz\": " << csrMatrix->getNNZ() << "\n";
    } else {
        ss << "    \"name\": \"" << matrixName << "\",\n";
        ss << "    \"rows\": 0,\n";
        ss << "    \"cols\": 0,\n";
        ss << "    \"nnz\": 0\n";
    }
    ss << "  },\n";

    if (!sequential) {
        ss << "  \"scenario\": {\n";
        ss << "    \"threads\": " << numThreads << ",\n";
        ss << "    \"scheduling_type\": \"" << schedulingType << "\",\n";
        ss << "    \"chunk_size\": " << chunkSize << "\n";
        ss << "  },\n";
    }

    ss << "  \"statistics90\": {\n";
    ss << "    \"duration_ms\": " << duration90 << ",\n";
    ss << "    \"FLOPs\": " << flops << ",\n";
    ss << "    \"GFLOPS\": " << gflops << ",\n";
    ss << "    \"Bandwidth_GBps\": " << bandwidthGBps << ",\n";
    ss << "    \"Arithmetic_intensity\": " << arithmeticIntensity << "\n";
    ss << "  },\n";
    ss << "  \"warmUp_time_ms\": " << warmupDuration << ",\n";

    ss << "  \"all_iteration_times_ms\": [";
    for (size_t i = 0; i < iterationDurations.size(); i++) {
        ss << iterationDurations[i];
        if (i != iterationDurations.size() - 1) ss << ", ";
    }
    ss << "],\n";

    // Errors
    ss << "  \"errors\": [";
    for (size_t i = 0; i < errors.size(); ++i) {
        ss << "\"" << errors[i] << "\"";
        if (i != errors.size() - 1) ss << ", ";
    }
    ss << "]\n";

    ss << "}";
    return ss.str();
}

void ResultsManager::clear() {
    iterationDurations.clear();
    errors.clear();
    warmupDuration = 0.0;
    duration90 = 0.0;
    flops = 0.0;
    gflops = 0.0;
    bandwidthGBps = 0.0;
}