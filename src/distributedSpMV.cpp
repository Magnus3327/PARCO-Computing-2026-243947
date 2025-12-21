#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <mpi.h>
#include <cstddef> // per offsetof

// Include i tuoi header esistenti
#include "CSR/CSRMatrix.h"
#include "MTX/MTXReader.h"
#include "ResultsManager/ResultsManager.h"
#include "Utils/Utils.h"

using namespace std;
using namespace mtx;

// 1. Definizione Datatype MPI per la struct Entry [cite: 332]
MPI_Datatype create_mpi_entry_type() {
    MPI_Datatype type;
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};

    displacements[0] = offsetof(Entry, row);
    displacements[1] = offsetof(Entry, col);
    displacements[2] = offsetof(Entry, value);

    MPI_Type_create_struct(3, blocklengths, displacements, types, &type);
    MPI_Type_commit(&type);
    return type;
}

// 2. Funzione SpMV Locale (adattata da D1) [cite: 242, 259]
void SpMV_Local(const CSRMatrix& localCSR, const double* fullX, double* localY) {
    for (int i = 0; i < localCSR.getRows(); i++) {
        double sum = 0.0;
        for (int j = localCSR.getIndexPointers(i); j < localCSR.getIndexPointers(i + 1); j++) {
            // Utilizziamo l'indice di colonna globale per accedere al vettore x completo [cite: 255]
            sum += localCSR.getData(j) * fullX[localCSR.getIndices(j)];
        }
        localY[i] = sum;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); [cite: 331]

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    MPI_Datatype MPI_ENTRY_TYPE = create_mpi_entry_type();
    
    // Parametri CLI semplificati
    if (argc < 2) {
        if (rank == 0) cerr << "Usage: mpirun -np <P> " << argv[0] << " <matrix_path> [-I=iterations]" << endl;
        MPI_Finalize();
        return 1;
    }

    string filePath = argv[1];
    int iterations = 1;
    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        if (arg.rfind("-I=", 0) == 0) iterations = stoi(arg.substr(3));
    }

    try {
        CSRMatrix localCSR;
        int globalRows = 0, globalCols = 0, globalNNZ = 0;
        vector<Entry> myEntries;

        // --- FASE 1: Lettura e Distribuzione Ciclica (Slide 14-15) ---
        if (rank == 0) { [cite: 195]
            vector<Entry> allEntries = readMTX(filePath); [cite: 195, 201]
            // Assumiamo che readMTX o un helper ci dia le dimensioni globali
            // Qui per brevità usiamo le info dalle entries
            for(const auto& e : allEntries) {
                globalRows = max(globalRows, e.row + 1);
                globalCols = max(globalCols, e.col + 1);
            }
            globalNNZ = allEntries.size();

            // Broadcast dimensioni globali
            int dims[3] = {globalRows, globalCols, globalNNZ};
            MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);

            // Smistamento ciclico [cite: 212, 214]
            vector<vector<Entry>> bins(P);
            for (auto& e : allEntries) {
                int target = e.row % P; [cite: 214]
                // Trasformazione indice riga: globale -> locale (Formula Prof) [cite: 217, 241]
                e.row = (e.row - target) / P; 
                bins[target].push_back(e);
            }

            // Invio dati agli altri rank [cite: 196]
            for (int p = 1; p < P; p++) {
                int count = bins[p].size();
                MPI_Send(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(bins[p].data(), count, MPI_ENTRY_TYPE, p, 1, MPI_COMM_WORLD);
            }
            myEntries = bins[0];
        } else {
            int dims[3];
            MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
            globalRows = dims[0]; globalCols = dims[1]; globalNNZ = dims[2];

            int count;
            MPI_Status status;
            MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            myEntries.resize(count);
            MPI_Recv(myEntries.data(), count, MPI_ENTRY_TYPE, 0, 1, MPI_COMM_WORLD, &status);
        }

        // Costruzione CSR locale [cite: 239]
        localCSR.buildFromEntries(myEntries);
        int localRows = localCSR.getRows();

        // --- FASE 2: Gestione Vettore x (Baseline Allgather) [cite: 18, 258] ---
        vector<double> fullX(globalCols);
        if (rank == 0) {
            // In un caso reale useresti generateRandomVector
            for(int i=0; i<globalCols; ++i) fullX[i] = (double)rand()/RAND_MAX;
        }
        // Ogni rank riceve la copia completa del vettore x 
        MPI_Bcast(fullX.data(), globalCols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // --- FASE 3: Calcolo SpMV Iterativo e Timing [cite: 299] ---
        vector<double> localY(localRows);
        ResultsManager rm; // Idealmente adattato per MPI

        // Warm-up [cite: 5]
        SpMV_Local(localCSR, fullX.data(), localY.data());

        double startTime = MPI_Wtime(); [cite: 331]
        for (int i = 0; i < iterations; i++) {
            SpMV_Local(localCSR, fullX.data(), localY.data());
        }
        double endTime = MPI_Wtime();

        double totalTime = (endTime - startTime) / iterations;
        double maxTime;
        // Riduzione per trovare il tempo del rank più lento (per lo speedup) [cite: 300]
        MPI_Reduce(&totalTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // --- FASE 4: Output Risultati (Solo Rank 0) [cite: 297] ---
        if (rank == 0) {
            // Calcolo FLOPS: 2 * NNZ_globali [cite: 301]
            double flops = (2.0 * globalNNZ) / maxTime; 
            cout << "{ \"matrix\": \"" << filePath << "\", \"processes\": " << P 
                 << ", \"avg_time_sec\": " << maxTime 
                 << ", \"GFLOPS\": " << (flops / 1e9) << " }" << endl;
        }

    } catch (const exception& e) {
        cerr << "Rank " << rank << " error: " << e.what() << endl;
    }

    MPI_Type_free(&MPI_ENTRY_TYPE);
    MPI_Finalize(); [cite: 331]
    return 0;
}