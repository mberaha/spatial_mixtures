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
    int numSamples = 10;
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

    std::deque<UnivariateState> chains;
    // Collector<UnivariateState> collector(1000);

    SpatialMixtureSampler spSampler(data);
    spSampler.init();

    for (int i=0; i < 1000; i++) {
        spSampler.sample();
    }

    for (int i=0; i < 100; i++) {
        spSampler.sample();
        if (i % 10 == 0) {
            std::cout << "Saving state" << std::endl;
            chains.push_back(spSampler.getStateAsProto());
        }
    }

    // spSampler.printDebugString();
    writeManyToFile(chains, "chains_now.dat");
    std::cout << "Done" << std::endl;


    std::deque<UnivariateState> restored;
    restored = readManyFromFile<UnivariateState>("chains_now.dat");

    std::cout << "******** RESTORED ********" << std::endl;
    UnivariateState state = restored[2];
    state.PrintDebugString();

    return 1;
}
