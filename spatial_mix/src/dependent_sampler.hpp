#ifndef SRC_DEPENDENT_SAMPLER_HPP
#define SRC_DEPENDENT_SAMPLER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <random>
#include <stan/math/prim/mat.hpp>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>

#include "PolyaGammaHybrid.h"
#include "collector.hpp"
#include "mcmc_utils.hpp"
#include "sampler_params.pb.h"
#include "univariate_mixture_state.pb.h"
#include "utils.hpp"

class DependentSpatialMixtureSampler {
 protected:
  // params
  SamplerParams params;

  // data
  int numGroups;
  std::vector<int> samplesPerGroup;
  std::vector<std::vector<double>> data;
  std::vector<Eigen::MatrixXd> predictors;
  int numdata;
  Eigen::MatrixXd W_init;

  // mixtures
  int numComponents;
  std::vector<Eigen::VectorXd> betas;
  std::vector<double> stddevs;

  Eigen::MatrixXd weights;  // one set of weights per location
  Eigen::MatrixXd transformed_weights;
  std::vector<std::vector<int>> cluster_allocs;
  Eigen::MatrixXd mtildes;
  int num_connected_comps;

  // MCAR
  double rho;
  Eigen::MatrixXd Sigma;
  Eigen::MatrixXd W;
  Eigen::MatrixXd SigmaInv;
  Eigen::MatrixXd F;
  std::vector<int> node2comp;
  double mtilde_sigmasq = 1.0;
  std::vector<std::vector<int>> comp2node;
  std::vector<Eigen::MatrixXd> F_by_comp;
  std::vector<Eigen::MatrixXd> G_by_comp;

  // Regression
  int p_size;
  Eigen::VectorXd reg_coeff;
  Eigen::VectorXd beta0;
  Eigen::MatrixXd beta_prec;
  Eigen::VectorXd reg_data;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> V;

  // prior for Sigma
  double nu;
  Eigen::MatrixXd V0;

  // prior for Rho
  double alpha;
  double beta;

  // adaptive MCMC for rho
  double sigma_n_rho;
  double rho_sum;
  double rho_sum_sq;
  int iter = 0;

  std::vector<Eigen::VectorXd> pippo;
  Eigen::MatrixXd sigma_star_h;
  // HyperParams for InverseGamma
  double priorA, priorB;

  unsigned long seed = 213513435;
  PolyaGammaHybridDouble *pg_rng = nullptr;
  std::mt19937_64 rng{213513435};

  // diagnostic for the MH sampler
  int numAccepted = 0;

 public:
  DependentSpatialMixtureSampler() {}

  DependentSpatialMixtureSampler(const SamplerParams &_params,
                                 const std::vector<std::vector<double>> &_data,
                                 const Eigen::MatrixXd &W,
                                 const std::vector<Eigen::MatrixXd> &X);

  ~DependentSpatialMixtureSampler() { delete (pg_rng); }

  void init();

  void sample();

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

  void _computeInvSigmaH();

  /*
   * Sampler the hyperparameters in the base measure P_0
   */
  void sampleHyperParams();

  void sample_mtilde();

  void saveState(Collector<DependentState> *collector);

  DependentState getStateAsProto();

  void printDebugString();

  const int &getNumAccepted() { return numAccepted; }
};

#endif  // SRC_SAMPLER_HPP
