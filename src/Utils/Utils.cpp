/*
    Utils.cpp
    
    This source file implements utility functions
*/

#include "Utils.h"

namespace utils {

    // Helper: create a static random engine that is initialized only once
    mt19937& getRandomEngine() {
        // Static ensures only one instance, seeded once
        static random_device rd; // hardware seed
        static mt19937 gen(rd()); 
        return gen;
    }

    double* generateRandomVector(int size, double minVal, double maxVal) {
        double* arr = new double[size];
        std::uniform_real_distribution<double> dist(minVal, maxVal);
        std::mt19937& gen = getRandomEngine();

        for(int i = 0; i < size; ++i) {
            arr[i] = dist(gen);
        }

        return arr; 
    }


    vector<Entry> generateMatrixEntries(int rows, int cols, double density) {
        if (rows <= 0 || cols <= 0)
            throw runtime_error("Invalid matrix dimensions");

        if (density <= 0.0 || density > 1.0)
            throw runtime_error("Density must be in (0, 1]");

        const size_t totalCells = static_cast<size_t>(rows) * cols;
        const size_t nnz = static_cast<size_t>(totalCells * density);

        vector<Entry> entries;
        entries.reserve(nnz);

        // RNG: fixed seed for reproducibility (importante per il benchmarking)
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> rowDist(0, rows - 1);
        std::uniform_int_distribution<int> colDist(0, cols - 1);
        std::uniform_real_distribution<double> valDist(-1.0, 1.0);

        // Set per evitare duplicati (row, col)
        unordered_set<size_t> used;
        used.reserve(nnz);

        while (entries.size() < nnz) {
            int r = rowDist(gen);
            int c = colDist(gen);

            // Chiave univoca per la coppia (r, c)
            size_t key = static_cast<size_t>(r) * cols + c;

            if (used.insert(key).second) {
                // Utilizziamo un initializer list per pulizia
                entries.push_back({r, c, valDist(gen)});
            }
        }

        // --- SORTING (Fondamentale per la conversione COO -> CSR) ---
        // Ordiniamo per riga, poi per colonna
        sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return (a.row == b.row) ? (a.col < b.col) : (a.row < b.row);
        });

        return entries;
    }

} // namespace utils