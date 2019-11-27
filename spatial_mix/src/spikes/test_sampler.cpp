#include <vector>
#include <iostream>
#include <stan/math.hpp>
#include "../sampler.hpp"
#include <random>


int main() {
    std::cout << "Beginning" << std::endl;
    std::mt19937_64 rng;
    int numGroups = 2;
    int numSamples = 200;
    std::vector<std::vector<double>> data(numGroups);
    for (int i=0; i < numGroups; i++) {
        data[i].resize(numSamples);
        for (int j=0; j < numSamples; j++) {
            double u = stan::math::uniform_rng(0.0, 1.0, rng);
            if (u < 0.3)
                data[i][j] = stan::math::normal_rng(0.0, 1.0, rng);
            else if(u < 0.6)
                data[i][j] = stan::math::normal_rng(-2.0, 1.0, rng);
            else
                data[i][j] = stan::math::normal_rng(5.0, 1.0, rng);
        }
    }

    std::cout << "Initialized Data" << std::endl;
    for (int i=0; i < numGroups; i++) {
        for (double d: data[i])
            std::cout << d << ", ";
        std::cout << "\n\n";
    }

    std::cout << data[0].size() << std::endl;

    SpatialMixtureSampler spSampler(data);
    spSampler.init();

    for (int i=0; i < 1000; i++)
        spSampler.sample();

    spSampler.printDebugString();

    return 1;
}
