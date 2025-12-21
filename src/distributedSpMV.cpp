#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <mpi.h>

// Include project headers
#include "CSR/CSRMatrix.h"
#include "MTX/MTXReader.h"
#include "ResultsManager/ResultsManager.h"
#include "Utils/Utils.h"

using namespace std;
using namespace utils;
using namespace mtx;

#include <mpi.h>
#include <iostream>
#include <string>
#include <stdexcept>

struct CLIOptions {
    std::string matrixPath = "";
    bool useGeneratedMatrix = false;
    int genSize = 0;
    int iterations = 1;
};

CLIOptions parseCLI(int argc, char* argv[], int rank) {
    CLIOptions opts;
    bool matrixProvided = false;
    std::string errorMsg = "";

    try {
        if (argc < 2) {
            throw std::runtime_error("Usage: mpirun -n <p> " + std::string(argv[0]) + " -M=path | -VM=size [-I=iterations]");
        }

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("-M=", 0) == 0) {
                if (opts.useGeneratedMatrix) throw std::logic_error("Conflict: -M and -VM");
                opts.matrixPath = arg.substr(3);
                matrixProvided = true;
            } 
            else if (arg.rfind("-VM=", 0) == 0) {
                if (matrixProvided) throw std::logic_error("Conflict: -M and -VM");
                opts.genSize = std::stoi(arg.substr(4));
                if (opts.genSize <= 0) throw std::out_of_range("Size must be > 0");
                opts.useGeneratedMatrix = true;
                matrixProvided = true;
            } 
            else if (arg.rfind("-I=", 0) == 0) {
                opts.iterations = std::stoi(arg.substr(3));
                if (opts.iterations <= 0) throw std::out_of_range("Iterations must be > 0");
            } 
            else {
                throw std::invalid_argument("Unknown: " + arg);
            }
        }

        if (!matrixProvided) throw std::runtime_error("Provide -M or -VM");

    } catch (const std::exception& e) {
        // Solo il rank 0 riporta l'errore per pulizia nei log
        if (rank == 0) {
            std::cerr << "\n[Rank 0] CLI ERROR: " << e.what() << std::endl;
        }
        // In MPI non basta lanciare l'eccezione, bisogna uscire tutti
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return opts;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    
}
