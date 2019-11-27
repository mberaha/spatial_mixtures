#include <vector>
#include <iostream>
#include <stan/math.hpp>
#include "../sampler.hpp"
#include "../collector.hpp"
#include <random>
#include "univariate_mixture_state.pb.h"


int main() {
    std::cout << "Beginning" << std::endl;
    std::mt19937_64 rng;
    int numGroups = 2;
    int numSamples = 50;
    std::cout << "numSamples: " << numSamples << std::endl;
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

    Collector<UnivariateState> collector(1000);

    SpatialMixtureSampler spSampler(data);
    spSampler.init();

    for (int i=0; i < 1000; i++) {
        spSampler.sample();
    }

    for (int i=0; i < 1000; i++) {
        spSampler.sample();
        if (remainder(i, 10) == 0)
            spSampler.saveState(&collector);
    }

    // spSampler.printDebugString();
    std::cout << "Calling collector.saveToFile" << std::endl;
    collector.saveToFile("chains.dat");
    std::cout << "Done" << std::endl;


    Collector<UnivariateState> collector2(1000);
    collector2.loadFromFile("chains.dat");

    std::cout << "******** RESTORED ********" << std::endl;
    UnivariateState state = collector2.get(1);
    std::cout << "State: " << std::endl;
    std::cout << state.num_components() << std::endl;
    state.PrintDebugString();

    return 1;
}
