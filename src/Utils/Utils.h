#ifndef UTILS_H
#define UTILS_H

#include <random>

using namespace std;

namespace utils {

    // Generate a random double vector of given size in [minVal, maxVal]
    // The random engine is initialized only once for consistent randomness
    double* generateRandomVector(int size, double minVal = 0.0, double maxVal = 1.0);

} // namespace utils

#endif // UTILS_H