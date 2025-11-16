#ifndef MTXREADER_H
#define MTXREADER_H

#include <vector>
#include <string>

using namespace std;

namespace mtx {

    struct Entry {
        int row, col;
        double value;
    };

    vector<Entry> readMTX(const string& filePath);

} // namespace mtx

#endif // MTXREADER_H