#include <vector>
#include <iostream>
#include <fstream>
#include <stan/math/prim/mat.hpp>

#include "../hdp.hpp"
#include "../recordio.hpp"
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

    std::deque<HdpState> chains;
    // Eigen::MatrixXd W = Eigen::MatrixXd::Zero(3, 3);
    HdpSampler sampler(data);
    std::cout<<"Init start"<<std::endl;
    sampler.init();
    sampler.check();
    std::cout<<"Init done"<<std::endl;
    // spSampler.printDebugString();
    for (int i=0; i < 5000; i++) {
        sampler.sample();
    }
    // spSampler.printDebugString();
    for (int i=0; i < 10000; i++) {
        sampler.sample();
        if ((i+1) % 10 == 0) {
            chains.push_back(sampler.getStateAsProto());
        }
    }

    // spSampler.printDebugString();
    writeManyToFile(chains, "chains_hdp.dat");
    std::cout << "Done" << std::endl;


    std::deque<HdpState> restored;
    restored = readManyFromFile<HdpState>("chains_now2.dat");

    std::cout << "******** RESTORED ********" << std::endl;

    return 1;
}
