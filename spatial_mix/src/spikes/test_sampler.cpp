#include <vector>
#include <iostream>
#include <fstream>
#include <stan/math/prim/mat.hpp>
#include "../sampler.hpp"
#include "../collector.hpp"
#include <random>
#include "univariate_mixture_state.pb.h"


int main() {
    std::cout << "Beginning" << std::endl;
    std::mt19937_64 rng;
    int numGroups = 3;
    int numSamples = 100;
    std::cout << "numSamples: " << numSamples << std::endl;
    std::vector<std::vector<double>> data(numGroups);
    for (int i=0; i < numGroups; i++) {
        data[i].resize(numSamples);
        for (int j=0; j < numSamples; j++) {
            double u = stan::math::uniform_rng(0.0, 1.0, rng);
            if(u < 0.5)
                data[i][j] = stan::math::normal_rng(-2.5, 1.0, rng);
            else
                data[i][j] = stan::math::normal_rng(2.5, 1.0, rng);
        }
    }

    std::ofstream outfile;
    outfile.open("data.csv");
    for (int i = 0; i < numGroups; i++) {
        for (int j=0; j < numSamples; j++) {
            outfile << i << "," << data[i][j] << std::endl;
        }
    }


    std::cout << "Initialized Data" << std::endl;
    for (int i=0; i < numGroups; i++) {
        for (double d: data[i])
            std::cout << d << ", ";
        std::cout << "\n\n";
    }

    std::deque<UnivariateState> chains;
    Eigen::MatrixXd W(3, 3);
    W << 0, 1, 1, 1, 0, 1, 1, 1, 0;
    // Eigen::MatrixXd W = Eigen::MatrixXd::Zero(3, 3);
    SpatialMixtureSampler spSampler(data, W);
    std::cout<<"Init start"<<std::endl;
    spSampler.init();
    std::cout<<"Init done"<<std::endl;
    spSampler.printDebugString();

    for (int i=0; i < 500; i++) {
        spSampler.sample();
    }
    spSampler.printDebugString();
    for (int i=0; i < 5; i++) {
        spSampler.sample();
        if ((i+1) % 10 == 0) {
            chains.push_back(spSampler.getStateAsProto());
        }
    }

    spSampler.printDebugString();
    writeManyToFile(chains, "chains_now2.dat");
    std::cout << "Done" << std::endl;


    std::deque<UnivariateState> restored;
    restored = readManyFromFile<UnivariateState>("chains_now2.dat");

    std::cout << "******** RESTORED ********" << std::endl;

    std::cout << "Acceptance rate for Rho: " <<
        1.0 * spSampler.getNumAccepted() / (1.0 * 15000) << std::endl;

    return 1;
}
