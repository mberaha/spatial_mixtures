#ifndef SRC_SAMPLER_HPP
#define SRC_SAMPLER_HPP

#include <random>
#include <vector>
#include "collector.hpp"
#include "protos/cpp/univariate_mixture_state.pb.h"
#include "PolyaGammaHybrid.h"

unsigned long int seed = 25112019;

class Sampler {
 protected:
     // data
     int numGroups;
     int samplesPerGroup;
     std::vector<std::vector<double>> data;

     // mixtures
     int numComponents;
     std::vector<double> means;
     std::vector<double> stddevs;

     std::vector<std::vector<double>> weights; // one set of weights per location
     std::vector<std::vector<double>> transformed_weights;

     // MCAR
     double rho;
     Eigen::MatrixXd Sigma;

     Collector<UnivariateMixtureState>* collector;

     PolyaGammaHybridDouble pg_rng(seed);
     std::mt19937_64 rng;

 public:
     Sampler(
        std::vector<std::vector<double>> data,
        Collector<UnivariateMixtureState>* collector);

    ~Sampler() {}

    void sample();

    void sampleAtoms();

    void sampleAllocations();

    void sampleWeights();

    void sampleRho();

    void sampleSigma();

    void sampleHyperParams();
}

#endif // SRC_SAMPLER_HPP
