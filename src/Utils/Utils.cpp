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

        return arr; // ricorda di fare delete[] dopo l'uso!
    }

} // namespace utils