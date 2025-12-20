/*
    MTXReader.cpp

    Implementation of a Matrix Market reader for .mtx files.
    It extracts non-zero entries into a vector of Entry structs, 
    each containing row index, column index, and value.

    Notes:
    - Row and column indices are converted from 1-based to 0-based.
    - Entries are sorted first by row, then by column.
    - The functionality is placed inside the 'mtx' namespace, 
      so no object instantiation is required.
    - Error handling is included for file operations and data parsing.
*/

#include "MTXReader.h"

namespace mtx {
    vector<Entry> readMTX(const string& filePath) {
        ifstream file(filePath);
        if (!file.is_open()) 
            throw runtime_error("Cannot open file: " + filePath);

        string line;
        vector<Entry> entries;

        // Skip Comments (% lines)
        while (getline(file, line)) {
            if (line[0] != '%') break;
        }

        // Check if there is a dimension line
        if (line.empty()) 
            throw runtime_error("Missing dimension line in " + filePath);

        int rows, cols, nnz;
        istringstream iss(line);

        // Check errors in reading dimensions
        if (!(iss >> rows >> cols >> nnz)) 
            throw runtime_error("Failed to read matrix dimensions.");
        if (rows <= 0 || cols <= 0 || nnz <= 0) 
            throw runtime_error("Invalid matrix dimensions.");

        entries.reserve(nnz);

        // Populate entries
        int row, col;
        double value;
        while (file >> row >> col >> value) 
            entries.push_back({row - 1, col - 1, value}); // switch from 1 based to 0-based

        // Sort by row then by column
        sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return (a.row == b.row) ? (a.col < b.col) : (a.row < b.row);
        });

        if (entries.empty())
            throw runtime_error("No entries read from file: " + filePath);

        return entries;
    } 
} // namespace mtx