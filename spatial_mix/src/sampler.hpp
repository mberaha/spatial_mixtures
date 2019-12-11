#ifndef SRC_SAMPLER_HPP
#define SRC_SAMPLER_HPP

#include <algorithm>
#include <random>
#include <vector>
#include "collector.hpp"
#include "univariate_mixture_state.pb.h"
#include "PolyaGammaHybrid.h"
#include <stan/math.hpp>
#include "utils.hpp"


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

     Eigen::MatrixXd weights; // one set of weights per location
     Eigen::MatrixXd transformed_weights;
     std::vector<std::vector<int>> cluster_allocs;

     // MCAR
     double rho;
     Eigen::MatrixXd Sigma;
     Eigen::MatrixXd W;

     // prior for Sigma
     double nu;
     Eigen::MatrixXd V0;

     std::vector<Eigen::VectorXd> pippo;
     std::vector<double> sigma_star_h;
     // HyperParams for NormalGamma
     double priorMean, priorA, priorB, priorLambda;

     unsigned long seed = 25112019;
     PolyaGammaHybridDouble* pg_rng;
     std::mt19937_64 rng{25112019};

 public:
     SpatialMixtureSampler(const std::vector<std::vector<double>> &_data,
                           const Eigen::MatrixXd &W);

    ~SpatialMixtureSampler() {
        delete(pg_rng);
    }

    void init();

    void sample() {
        sampleAtoms();
        sampleAllocations();
        sampleSigma();
    }

    /*
     * We use a Normal kernel with conjugate Normal - Inverse Gamma
     * base measure, so the update of the atom is
     * Law(\mu, \sigma)_h \propto
     *  P_0(\mu, \sigma) \prod_{(i, j): s_{ij}=h} N(y_{ij} | \mu_h, \sigma_h)
     * That is a conjugate normal likelihood with normal inverse gamma prior
     */
    void sampleAtoms();

    void sampleAllocations();

    /*
     * We use the PolyaGamma trick to sample from the transformed weights
     */
    void sampleWeights();

    /*
     * This step requires a Metropolis Hastings step
     */
    void sampleRho();

    /*
     * We use a conjugate Inverse - Wishart prior for Sigma, so the
     * posterior law of Sigma is still Inverse - Wishart with parameters
     * (\nu + I, Psi + \sum_{i=1}^I (tw_i - \mu_i) (tw_i - \mu_i)^T
     * for tw = transformed weights
     * \mu_i = \rho N^{-1} \sum{j \n N(i)} tw_j
     */
    void sampleSigma();

    void _computeInvSigmaH();

    /*
     * Sampler the hyperparameters in the base measure P_0
     */
    void sampleHyperParams();

    void sampleNumComponents();

    void saveState(Collector<UnivariateState>* collector);

    UnivariateState getStateAsProto();

    void printDebugString();

    std::vector<double> _normalGammaUpdate(std::vector<double> data);
};

#endif // SRC_SAMPLER_HPP
