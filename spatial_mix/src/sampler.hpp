#ifndef SRC_SAMPLER_HPP
#define SRC_SAMPLER_HPP

#include <algorithm>
#include <random>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
// #include <cstdlib>
#include "collector.hpp"
#include "univariate_mixture_state.pb.h"
#include "sampler_params.pb.h"
#include "PolyaGammaHybrid.h"
#include <stan/math/prim/mat.hpp>
#include "mcmc_utils.hpp"
#include "utils.hpp"


class SpatialMixtureSampler {
 protected:
     // params
     SamplerParams params;

     // data
     int numGroups;
     std::vector<int> samplesPerGroup;
     std::vector<std::vector<double>> data;
     int numdata;
     Eigen::MatrixXd W_init;

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
     Eigen::MatrixXd SigmaInv;
     Eigen::MatrixXd F;

     // Boundary Detection
     int numMetrics; // numero di metriche di dissimiglianza tra le locations
     int coin_toss; // mi salvo l'esito della bernoulli del boundary
     std::vector<std::vector<int>> neigh; // per ogni location i suoi potenziali
     //da dare in input perchè in python è easy.
     std::vector <Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>> diss_metrics;
     // vettore di matrici: ogni posto del vettore contiene la dissimilarity
     // matrix della metrica k-esima
     Eigen::VectorXd bound_coeff; // gli alpha
     Eigen::MatrixXd p_bern; // salvo i parametri delle bernoulli dei boundary
     Eigen::MatrixXd B; //matrice dei boundaries
     Eigen::MatrixXd aux_boundary; // matrice ausiliaria per la matrix norm
     std::vector<std::vector<Eigen::VectorXd>> pairwise_diss;


     // Regression
     bool regression = false;
     int p_size;
     Eigen::VectorXd reg_coeff;
     Eigen::MatrixXd predictors;
     Eigen::VectorXd reg_coeff_mean;
     Eigen::MatrixXd reg_coeff_prec;
     Eigen::VectorXd reg_data;
     Eigen::VectorXd mu;
     Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> V;

     // prior for Sigma
     double nu;
     Eigen::MatrixXd V0;

     // prior for Rho
     double alpha;
     double beta;

     std::vector<Eigen::VectorXd> pippo;
     std::vector<double> sigma_star_h;
     // HyperParams for NormalGamma
     double priorMean, priorA, priorB, priorLambda;

     unsigned long seed = 213513435;
     PolyaGammaHybridDouble* pg_rng = nullptr;
     std::mt19937_64 rng{213513435};

     // diagnostic for the MH sampler
     int numAccepted = 0;

     bool verbose = false;

 public:
     SpatialMixtureSampler() {}

     SpatialMixtureSampler(
        const SamplerParams &_params,
        const std::vector<std::vector<double>> &_data,
        const Eigen::MatrixXd &W,
        const std::vector <Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>> &metrics,
        const std::vector<std::vector<int>> &_neigh);

    SpatialMixtureSampler(
       const SamplerParams &_params,
       const std::vector<std::vector<double>> &_data,
       const Eigen::MatrixXd &W,
       const std::vector <Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>> &metrics,
       const std::vector<std::vector<int>> &_neigh,
       const std::vector<Eigen::MatrixXd> &X);

    ~SpatialMixtureSampler() {
        delete(pg_rng);
    }

    void init();

    void sample();
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

    void regress();

    void computeRegressionResiduals();

    void sampleB();

    void sampleBoundaryCoeffs();

    void _computeInvSigmaH();

    /*
     * Sampler the hyperparameters in the base measure P_0
     */
    void sampleHyperParams();

    void sampleNumComponents();

    void saveState(Collector<UnivariateState>* collector);

    UnivariateState getStateAsProto();

    void printDebugString();

    void set_verbose() { verbose = true;}

    void unset_verbose() {verbose = false;}

    const int& getNumAccepted() {return numAccepted;}
};

#endif // SRC_SAMPLER_HPP
