#include "BenchmarkResult.h"
#include <sstream>

BenchmarkResult::BenchmarkResult() {}
BenchmarkResult::~BenchmarkResult() {
    clear();
}

void BenchmarkResult::addResult(const CSRMatrix& csr, int numThreads, string schedulingType, int chunkSize, double duration, const string pathmatrix) {
    string matrixName = pathmatrix.substr(pathmatrix.find_last_of("/\\") + 1); 
    
    stringstream ss;
    ss << "{\n";
    ss << "  \"threads\": " << numThreads << ",\n";
    ss << "  \"matrix\": {\n";
    ss << "    \"name\": \"" << matrixName << "\",\n";
    ss << "    \"rows\": " << csr.getRows() << ",\n";
    ss << "    \"cols\": " << csr.getCols() << ",\n";
    ss << "    \"nnz\": " << csr.getNNZ() << "\n";
    ss << "  },\n";
    ss << "  \"scenario\": {\n";
    ss << "    \"scheduling_type\": \"" << schedulingType << "\",\n";
    ss << "    \"chunk_size\": \"" << (chunkSize == 0 ? "default" : std::to_string(chunkSize)) << "\"\n";
    ss << "  },\n";
    ss << "  \"duration_milliseconds\": " << duration << "\n";
    ss << "}";

    results.push_back(ss.str());
}

void BenchmarkResult::addResult(CSRMatrix& csr, double duration, string pathmatrix) {
    string matrixName = pathmatrix.substr(pathmatrix.find_last_of("/\\") + 1); 
    
    stringstream ss;
    ss << "{\n";
    ss << "  \"matrix\": {\n";
    ss << "    \"name\": \"" << matrixName << "\",\n";
    ss << "    \"rows\": " << csr.getRows() << ",\n";
    ss << "    \"cols\": " << csr.getCols() << ",\n";
    ss << "    \"nnz\": " << csr.getNNZ() << "\n";
    ss << "  },\n";
    ss << "  \"duration_milliseconds\": " << duration << "\n";
    ss << "}";

    results.push_back(ss.str());
}


void BenchmarkResult::addError(const std::string& errorMessage) {
    errors.push_back(errorMessage);
}


string BenchmarkResult::toJSON() const {
    stringstream ss;
    ss << "{\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        ss << "    " << results[i];
        if (i != results.size() - 1) ss << ",";
        ss << "\n";
    }
    ss << "  ],\n  \"errors\": [\n";
    for (size_t i = 0; i < errors.size(); ++i) {
        ss << "    \"" << errors[i] << "\"";
        if (i != errors.size() - 1) ss << ",";
        ss << "\n";
    }
    ss << "  ]\n}";
    return ss.str();
}

void BenchmarkResult::clear() {
    results.clear();
    errors.clear();
}