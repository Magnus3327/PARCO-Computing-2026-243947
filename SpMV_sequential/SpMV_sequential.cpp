#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm> // for sort and max_element functions
#include <random>
#include <chrono>
#include <iomanip>  // for fixed and setprecision

using namespace std;
using namespace std::chrono;

struct CSRMatrix {
    int *indexPointers = nullptr;
    int *indices = nullptr;
    double *data = nullptr;
    int rows = 0, cols = 0, nnz = 0;
};

// Entry structure for MTX file
struct Entry {
    int row, col;
    double value;
};

vector<Entry> readSparseMTX(const string& filePath) {
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

    return entries;
}

// Build CSR from entries
void CSR(CSRMatrix& csr, const vector<Entry>& entries) {
    if (entries.empty())
        throw runtime_error("Cannot build CSR: entries vector is empty.");

    // Set CSR dimensions, it can be possible directly from readMTX function but i prefer to compute in a standalone way
    csr.nnz = entries.size();
    csr.rows = max_element(entries.begin(), entries.end(),
        [](const Entry& a, const Entry& b) { return a.row < b.row; })->row + 1;
    csr.cols = max_element(entries.begin(), entries.end(),
        [](const Entry& a, const Entry& b) { return a.col < b.col; })->col + 1;

    // Allocate CSR arrays
    csr.indexPointers = new int[csr.rows + 1]; // +1 for the end pointer
    csr.indices = new int[csr.nnz];
    csr.data = new double[csr.nnz];
    if (!csr.indexPointers || !csr.indices || !csr.data) 
        throw runtime_error("Memory allocation failed.");
    
    int currentRow = 0;
    csr.indexPointers[0] = 0;  // The first row always starts with 0

    for (int i = 0; i < entries.size(); i++) {
        const Entry& entry = entries[i];

        // new row -> fill all missing row pointers
        while (currentRow < entry.row) {
            currentRow++;
            csr.indexPointers[currentRow] = i;  // Marks where the next row starts
        }

        // Store the current nonzero element
        csr.data[i] = entry.value;
        csr.indices[i] = entry.col;
    }

    // Fill in the remaining row pointers (for empty rows or the last row)
    while (currentRow < csr.rows) {
        currentRow++;
        csr.indexPointers[currentRow] = csr.nnz;  
    }
}

// Print CSR matrix (DEBUG)
void printCSR(const CSRMatrix& csr) {
    cout << "\n"<< "CSR Representation:" << endl;
    cout << "Rows -> " << csr.rows << endl;
    cout << "Cols -> " << csr.cols << endl;
    cout << "NNZ -> " << csr.nnz  << endl;
    cout << endl;
    cout << "Index Pointers -> ";
    for (int i = 0; i <= csr.rows; i++)
        cout << csr.indexPointers[i] << " ";

    cout << "\n\n" << "Indices -> ";
    for (int i = 0; i < csr.nnz; i++) 
        cout << csr.indices[i] << " ";

    cout << "\n\n" << "Data -> ";
    for (int i = 0; i < csr.nnz; i++) 
        cout << csr.data[i] << " ";

    cout << "\n";
}

// Free memory safely
void freeCSR(CSRMatrix& csr) {
    delete[] csr.indexPointers;
    delete[] csr.indices;
    delete[] csr.data;
    csr.indexPointers = csr.indices = nullptr;
    csr.data = nullptr;
}

// Function to generate a random double vector
double* generateRandomVector(int size, double minVal = 0.0, double maxVal = 1.0) {
    double* v = new double[size];
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(minVal, maxVal);

    for (int i = 0; i < size; ++i)
        v[i] = dist(gen);

    return v;
}

/*Print results in json-like output for easy parsing into external scripts
{
    "matrix": {
        "name": matrixName,
        "rows": csr.rows,
        "cols": csr.cols,
        "nnz": csr.nnz
    },
    "duration_milliseconds": duration
}
*/
string generateJSONOutput(CSRMatrix& csr, double duration, string matrixName = "unknown") {

    stringstream ss;
    ss << "{\n";
    ss << "  \"matrix\": {\n";
    ss << "    \"name\": \"" << matrixName << "\",\n";
    ss << "    \"rows\": " << csr.rows << ",\n";
    ss << "    \"cols\": " << csr.cols << ",\n";
    ss << "    \"nnz\": " << csr.nnz << "\n";
    ss << "  },\n";
    ss << "  \"duration_milliseconds\": " << duration << "\n";
    ss << "}";
    return ss.str();
}

// Print the json with array of results and array of errors
void outputFinalJSON(const vector<string>& resultJsons, const vector<string>& errorMessages) {
    cout << "{\n";
    cout << "  \"results\": [\n";
    for (size_t i = 0; i < resultJsons.size(); ++i) {
        cout << resultJsons[i];
        if (i != resultJsons.size() - 1) cout << ",\n";
    }
    cout << "\n  ],\n";

    cout << "  \"errors\": [\n";
    for (size_t i = 0; i < errorMessages.size(); ++i) {
        cout << "    \"" << errorMessages[i] << "\"";
        if (i != errorMessages.size() - 1) cout << ",\n";
    }
    cout << "\n  ]\n";
    cout << "}\n";
}

double* SpMV(const CSRMatrix& csr, const double* x, double& duration) {
    double* y = new double[csr.rows];
    auto start = high_resolution_clock::now();

    //row-major SpMV to optimize cache usage
    for (int i = 0; i < csr.rows; i++) {
        double sum = 0.0;   //temporary sum to avoid multiple writes to y[i]
        for (int j = csr.indexPointers[i]; j < csr.indexPointers[i + 1]; j++) {
            sum += csr.data[j] * x[csr.indices[j]];
        }
        y[i] = sum;
    }

    auto end = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end - start).count() / 1e6;
    return y;
}

int main(int argc, char* argv[]) {
    vector<string> resultJsons = {};
    vector<string> errorMessages = {};

    // Collect and validate cli arguments
    if (argc < 2) {
        errorMessages.push_back(string(argv[0]) + " needs" + " matrix_path [-I=iterations]");
        outputFinalJSON(resultJsons, errorMessages);
        return 1;
    }
    
    string filePath = argv[1];
    string matrix = filePath.substr(filePath.find_last_of("/\\") + 1); // extract filename from path

    CSRMatrix csr;
    double *inputVector = nullptr;
    double *outputVector = nullptr;
    int iterations = 1;

    if(argc > 2) {
        string arg = argv[2];

        if (arg.rfind("-I=", 0) == 0) {
            string val = arg.substr(3);
            istringstream iss(val);
            int it = 0;
            if (!(iss >> it) || it < 1) {
                errorMessages.push_back( "Invalid iterations: '" + val + "'. Must be > 0.");
                outputFinalJSON(resultJsons, errorMessages);
                return 1;
            }
            iterations = it;
        } else {
            errorMessages.push_back("Unknown argument: '" + arg + "'.");
            outputFinalJSON(resultJsons, errorMessages);
            return 1;
        }
    }

    try {
        vector<Entry> entries = readSparseMTX(filePath);
        //cout << "Sparse Matrix successfully read (" << entries.size() << " non-zero entries)\n";
        CSR(csr, entries);

        // Generate input vector
        inputVector = generateRandomVector(csr.cols, -1000.0, 1000.0);  // Random values between -100 and 100
        double duration = 0.0;
        outputVector = SpMV(csr, inputVector, duration);

        // Actual timed SpMV executions
        for(int i=0; i<iterations; i++){
            outputVector = SpMV(csr, inputVector, duration);
            resultJsons.push_back(generateJSONOutput(csr, duration, matrix));
        }
        
        // print final JSON output
        outputFinalJSON(resultJsons, errorMessages);
    }
    catch (const exception& e) {
        errorMessages.push_back(string("Fatal error: ") + e.what());
        // Return here because no point continuing without matrix
        outputFinalJSON(resultJsons, errorMessages);
        return 1;
    }

    freeCSR(csr);
    delete[] inputVector;
    delete[] outputVector;
    return 0;
}