/*
    ResultsManager.h
    
    This class manages the results and errors of the runs,
    storing them and providing a method to output them in JSON format.

    The json format for result (parallel) is as follows:
    {
        "threads": num_threads,
        "matrix": {
            "name": matrixName,
            "rows": csr.getRows(),
            "cols": csr.getCols(),
            "nnz": csr.getNNZ()
        },
        "scenario": {
            "scheduling_type": schedulingType,
            "chunk_size": chunkSize
        },
        "duration_milliseconds": duration
    }

    The json format for result (sequential) is as follows:
    {
        "matrix": {
            "name": matrixName,
            "rows": csr.getRows(),
            "cols": csr.getCols(),
            "nnz": csr.getNNZ()
        },
        "duration_milliseconds": duration
    }

    The results and errors are stored in vectors of strings, where each string
    represents a JSON object for a single result or error message.

    The class provides methods to add results and errors, and a method to
    generate a complete JSON representation of all results and errors.
*/

#ifndef RESULTSMANAGER_H
#define RESULTSMANAGER_H

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include "CSR/CSRMatrix.h"

using namespace std;

class ResultsManager {
private:
    vector<string> results;
    vector<string> errors;

public:
    ResultsManager();
    ~ResultsManager();

    // Add result, its overloaded for parallel and sequential benchmarks
    void addResult(const CSRMatrix& csr, int num_threads, string schedulingType, int chunksize, double duration, string matrixName = "unknown");
    void addResult(CSRMatrix& csr, double duration, string matrixName = "unknown");

    // Add error message
    void addError(const string& errorMessage) ;

    // Generate JSON representation of results and errors in a string to be printed
    string toJSON() const ;

    // Remove all stored results and errors, called in the destructor
    void clear();
};

#endif // RESULTSMANAGER_H