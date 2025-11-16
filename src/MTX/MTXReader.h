/*
    MTXReader.h

    This header defines: a MTX file reader that parses Matrix Market files 
    and extracts the non-zero entries into a vector of Entry structs;
    an Entry struct contains the row index, column index, and value of a non-zero matrix element.

    I chose to implement this as a separate module to promote code reusability and separation of concerns.
    This way, any part of the project that needs to read MTX files can simply include this header and use its functionality
    without duplicating code or logic elsewhere in the project.

    I used a namespace instead of a class because the functionality is stateless and does not require object instantiation.
    A namespace is more appropriate for grouping related functions and types in this case.
*/

#ifndef MTXREADER_H
#define MTXREADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

using namespace std;

namespace mtx {

    struct Entry {
        int row, col;
        double value;
    };

    vector<Entry> readMTX(const string& filePath);

} // namespace mtx

#endif // MTXREADER_H