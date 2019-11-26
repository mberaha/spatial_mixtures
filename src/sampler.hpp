#ifndef SRC_SAMPLER_HPP
#define SRC_SAMPLER_HPP

#include <random>
#include <vector>
#include "collector.hpp"
#include "protos/cpp/univariate_mixture_state.pb.h"
#include "PolyaGammaHybrid.h"
#include <stan/math.hpp>


class SpatialMixtureSampler {
 protected:
     // data
     int numGroups;
     std::vector<int> samplesPerGroup;
     std::vector<std::vector<double>> data;
     int numdata;

     // mixtures
     int numComponents;
     std::vector<double> means;
     std::vector<double> stddevs;

     std::vector<Eigen::VectorXd> weights; // one set of weights per location
     std::vector<Eigen::VectorXd> transformed_weights;

     std::vector<std::vector<int>> cluster_allocs;

     // MCAR
     double rho;
     Eigen::MatrixXd Sigma;
     Eigen::MatrixXi W;

     // HyperParams for NormalGamma
     double priorMean, priorA, priorB, priorLambda;

     unsigned long seed = 25112019;
     PolyaGammaHybridDouble* pg_rng;
     std::mt19937_64 rng{25112019};

 public:
     SpatialMixtureSampler(const std::vector<std::vector<double>> &_data);

    ~SpatialMixtureSampler() {
        delete(pg_rng);
    }

    void init();

    void sample() {
        sampleAtoms();
        sampleAllocations();
    }

    void sampleAtoms();

    void sampleAllocations();

    void sampleWeights();

    void sampleRho();

    void sampleSigma();

    void sampleHyperParams();

    void sampleNumComponents();

    void saveState(Collector<UnivariateState>* collector);

    void printDebugString();

    std::vector<double> _normalGammaUpdate(std::vector<double> data);
};

#endif // SRC_SAMPLER_HPP
