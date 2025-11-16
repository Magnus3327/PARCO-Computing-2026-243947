#ifndef BENCHMARKRESULT_H
#define BENCHMARKRESULT_H

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include "CSR/CSRMatrix.h"

using namespace std;

class BenchmarkResult {
private:
    vector<string> results;
    vector<string> errors;

public:
    BenchmarkResult();
    ~BenchmarkResult();

    // Aggiunge un risultato
    void addResult(const CSRMatrix& csr, int num_threads, string schedulingType, int chunksize, double duration, string matrixName = "unknown");
    void addResult(CSRMatrix& csr, double duration, string matrixName = "unknown");

    // Aggiunge un messaggio di errore
    void addError(const string& errorMessage) ;

    // Genera JSON finale dei risultati e degli errori
    string toJSON() const ;

    // Rimuove tutti i dati
    void clear();
};

#endif // BENCHMARKRESULT_H