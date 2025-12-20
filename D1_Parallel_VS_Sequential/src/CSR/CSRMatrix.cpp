/*
    Implementation of CSRMatrix class for Compressed Sparse Row matrix representation.
*/

#include "./CSRMatrix.h"

// Constructor set members to zero/nullptr
CSRMatrix::CSRMatrix() : indexPointers(nullptr), indices(nullptr), data(nullptr), rows(0), cols(0), nnz(0) {}

CSRMatrix::~CSRMatrix() {
    clear();
}

void CSRMatrix::clear() {
    delete[] indexPointers;
    delete[] indices;
    delete[] data;
    indexPointers = nullptr;
    indices = nullptr;
    data = nullptr;
    rows = cols = nnz = 0;
}

// Build CSR from entries (row, col, value)
void CSRMatrix::buildFromEntries(const vector<Entry>& entries) {
    if (entries.empty())
        throw runtime_error("Cannot build CSR: entries vector is empty.");

    // For safety, clear any existing data
    clear();

    nnz = entries.size();
    rows = max_element(entries.begin(), entries.end(),
        [](const Entry& a, const Entry& b) { return a.row < b.row; })->row + 1;
    cols = max_element(entries.begin(), entries.end(),
        [](const Entry& a, const Entry& b) { return a.col < b.col; })->col + 1;

    // Allocate CSR arrays
    indexPointers = new int[rows + 1]; // +1 for the end pointer
    indices = new int[nnz];
    data = new double[nnz];
    if (!indexPointers || !indices || !data) 
        throw runtime_error("Memory allocation failed.");
    
    int currentRow = 0;
    indexPointers[0] = 0;  // The first row always starts with 0

    for (int i = 0; i < entries.size(); i++) {
        const Entry& entry = entries[i];

        // new row -> fill all missing row pointers
        while (currentRow < entry.row) {
            currentRow++;
            indexPointers[currentRow] = i;  // Marks where the next row starts
        }

        // Store the current nonzero element
        data[i] = entry.value;
        indices[i] = entry.col;
    }

    // Fill in the remaining row pointers (for empty rows or the last row)
    while (currentRow < rows) {
        currentRow++;
        indexPointers[currentRow] = nnz;  
    }
}

// Debug: print CSR
void CSRMatrix::print() const {
    cout << "\nCSR Matrix:\n";
    cout << "Rows: " << rows << " Cols: " << cols << " NNZ: " << nnz << "\n";

    cout << "IndexPointers: ";
    for (int i = 0; i <= rows; ++i) cout << indexPointers[i] << " ";
    cout << "\n";

    cout << "Indices: ";
    for (int i = 0; i < nnz; ++i) cout << indices[i] << " ";
    cout << "\n";

    cout << "Data: ";
    for (int i = 0; i < nnz; ++i) cout << data[i] << " ";
    cout << "\n";
}