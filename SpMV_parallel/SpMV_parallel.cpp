#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm> // for sort and max_element functions
#include <random>
#include <iomanip>  // for fixed and setprecision
#include <cstdlib> // for getenv to read environment variables
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

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

// Sparse Matrix-Vector Multiplication (SpMV) y = A * x in parallel using OpenMP
double* SpMV(const CSRMatrix& csr, const double* x, double& duration, string schedulingType = "static", int chunksize = 0) {
    double* y = new double[csr.rows];
    double start = 0.0, end = 0.0;

    #ifdef _OPENMP
    // Select scheduling policy
    // using runtime scheduling to allow dynamic selection
    omp_sched_t schedule;
    if(schedulingType == "static") schedule = omp_sched_static;
    else if(schedulingType == "dynamic") schedule = omp_sched_dynamic;
    else if(schedulingType == "guided")  schedule = omp_sched_guided;
    else throw runtime_error("Invalid scheduling type. Use static, dynamic, or guided.");
    // auto

    // Set OpenMP scheduling mode using the selected policy and chunk size
    // If chunksize is 0, OpenMP will use the default chunk size for the selected scheduling policy
    omp_set_schedule(schedule, chunksize);
    #endif

    #ifdef _OPENMP
    start = omp_get_wtime();
    #endif

    #pragma omp parallel for schedule(runtime) shared(csr, x, y)
    for (int i = 0; i < csr.rows; i++) {
        double sum = 0.0;  //temporary sum to avoid multiple writes to y[i]
        for (int j = csr.indexPointers[i]; j < csr.indexPointers[i + 1]; j++) {
            sum += csr.data[j] * x[csr.indices[j]];
        }
        y[i] = sum;
    }

    #ifdef _OPENMP
    end = omp_get_wtime();
    #endif

    duration = (end - start) * 1e3; // convert seconds to milliseconds
    return y;
}

/*Print results in json-like output for easy parsing into external scripts
{
    "threads": num_threads, 
    "matrix": {
        "name": matrixName,
        "rows": csr.rows,
        "cols": csr.cols,
        "nnz": csr.nnz
    },
    "scenario": {
        "scheduling_type": schedulingType,
        "chunk_size": chunksize
    },
    "duration_milliseconds": duration
}
*/
string generateJSONOutput(int num_threads, const CSRMatrix& csr, string schedulingType, int chunksize, double duration, string matrixName = "unknown") {

    stringstream ss;
    ss << "{\n";
    ss << "  \"threads\": " << num_threads << ",\n";
    ss << "  \"matrix\": {\n";
    ss << "    \"name\": \"" << matrixName << "\",\n";
    ss << "    \"rows\": " << csr.rows << ",\n";
    ss << "    \"cols\": " << csr.cols << ",\n";
    ss << "    \"nnz\": " << csr.nnz << "\n";
    ss << "  },\n";
    ss << "  \"scenario\": {\n";
    ss << "    \"scheduling_type\": \"" << schedulingType << "\",\n";
    ss << "    \"chunk_size\": \"" << (chunksize == 0 ? "default" : to_string(chunksize)) << "\"\n";
    ss << "  },\n";
    ss << "  \"duration_milliseconds\": " << duration << "\n";
    ss << "}";
    return ss.str();
}

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

int main(int argc, char* argv[]) {
    vector<string> resultJsons = {};
    vector<string> errorMessages = {};
    bool errorsCLI = false;

    // Collect and validate cli arguments
    if (argc < 2) {
        errorMessages.push_back(string(argv[0]) + " needs" + " matrix_path [-T=num_threads] [-S=scheduling] [-C=chunkSize] [-I=iterations]");
        outputFinalJSON(resultJsons, errorMessages);
        return 1;
    }

    CSRMatrix csr;
    
    double *inputVector = nullptr;
    double *outputVector = nullptr;

    string filePath = argv[1];
    string matrix = filePath.substr(filePath.find_last_of("/\\") + 1); // extract filename from path

    // Defaults
    string schedulingType = "static"; 
    int chunkSize = 0; 
    int iterations = 1;
    double duration = 0.0;
    int num_threads = 1;

    // Set default number of threads from OMP_NUM_THREADS if available
    // If not set, use omp_get_max_threads()
    // If the user specifies -T=, it will override this value
    #ifdef _OPENMP
    const char* omp_env = getenv("OMP_NUM_THREADS");
    if (omp_env != nullptr) {
        int env_threads = atoi(omp_env);
        num_threads = (env_threads > 0) ? env_threads : omp_get_max_threads();
    } else {
        num_threads = omp_get_max_threads();
    }
    #endif

    // Parse options starting from argv[2]
    // With this approach, the order of options does not matter
    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];

        // Threads
        if (arg.rfind("-T=", 0) == 0) {
            string val = arg.substr(3);
            istringstream iss(val);
            int threadN = 0;
            if (!(iss >> threadN) || threadN < 1) {
                errorMessages.push_back("Invalid number of threads: '" + val + "'");
                errorsCLI = true;
            }
            #ifdef _OPENMP
            int max_threads = omp_get_max_threads();
            if (threadN > max_threads) {
                errorMessages.push_back(
                        "Requested threads (" + to_string(threadN) + ") exceed max available (" + to_string(max_threads) + ")."
                    );
                errorsCLI = true;
            }
            #endif
            num_threads = threadN;
        } // Scheduling type
        else if (arg.rfind("-S=", 0) == 0) {
            schedulingType = arg.substr(3);
            if (schedulingType != "static" && schedulingType != "dynamic" && schedulingType != "guided") {
                errorMessages.push_back("Invalid scheduling type: '" + schedulingType + "'. Use static, dynamic, or guided.");
                errorsCLI = true;
            }
        } // Chunk size
        else if (arg.rfind("-C=", 0) == 0) {
            string val = arg.substr(3);
            istringstream iss(val);
            int cs = 0;
            if (!(iss >> cs) || cs < 0) {
                errorMessages.push_back("Invalid chunk size: '" + val + "'. Must be >= 0.");
                errorsCLI = true;
            }
            chunkSize = cs;
        } // Number of iterations
        else if (arg.rfind("-I=", 0) == 0) {
            string val = arg.substr(3);
            istringstream iss(val);
            int it = 0;
            if (!(iss >> it) || it < 1) {
                errorMessages.push_back( "Invalid iterations: '" + val + "'. Must be > 0.");
                errorsCLI = true;
            }
            iterations = it;
        }
        else {
            errorMessages.push_back("Unknown argument: '" + arg + "'.");
            errorsCLI = true;
        }
    }

    // if the are errors, print in json->errors[] and exit
    // if the matrix cannot be readed, the error will be caught in the try-catch below
    if(errorsCLI) {
        outputFinalJSON(resultJsons, errorMessages);
        return 1;
    }

    try {
        vector<Entry> entries = readSparseMTX(filePath);
        //cout << "Sparse Matrix successfully read (" << entries.size() << " non-zero entries)\n";
        
        CSR(csr, entries);
        //printCSR(csr);

        // Generate input vector
        inputVector = generateRandomVector(csr.cols, -1000.0, 1000.0);  // Random values between -1000 and 1000

        //warm-up phase to avoid measuring thread creation overhead
        outputVector = SpMV(csr, inputVector, duration, schedulingType, chunkSize);
        duration = 0.0;

        // Actual timed SpMV executions
        for(int i=0; i<iterations; i++){
            outputVector = SpMV(csr, inputVector, duration, schedulingType, chunkSize);
            resultJsons.push_back(generateJSONOutput(num_threads, csr, schedulingType, chunkSize, duration, matrix));
        }

        // print final JSON output
        outputFinalJSON(resultJsons, errorMessages);
    }
    catch (const exception& e) {
        //
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