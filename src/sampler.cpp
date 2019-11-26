#include "sampler.hpp"
#include <numeric>

using namespace stan::math;

SpatialMixtureSampler::SpatialMixtureSampler(
        const std::vector<std::vector<double>> &_data): data(_data) {
    numGroups = data.size();
    samplesPerGroup.resize(numGroups);
    for (int i=0; i < numGroups; i++) {
        samplesPerGroup[i] = data[i].size();
    }
    numdata = std::accumulate(
        samplesPerGroup.begin(), samplesPerGroup.end(), 0);
}

void SpatialMixtureSampler::init() {
    // TODO now we fix this, remember to put priors or pass from
    // constructor
    priorMean = 0.0;
    priorA = 2.0;
    priorB = 2.0;
    priorLambda = 10.0;
    rho = 0.9;
    numComponents = 10;

    Sigma = Eigen::MatrixXd::Identity(numComponents, numComponents);

    // Now proper initialization
    means.resize(numComponents);
    stddevs.resize(numComponents);
    weights.resize(numGroups);
    transformed_weights.resize(numGroups);
    cluster_allocs.resize(numGroups);
    for (int i=0; i < numGroups; i++) {
        weights[i].resize(numComponents);
        transformed_weights[i].resize(numComponents - 1);
        cluster_allocs[i].resize(samplesPerGroup[i]);
    }

    for (int h=0; h < numComponents; h++) {
        means[h] = stan::math::normal_rng(0.0, 10.0, rng);
        stddevs[h] = stan::math::uniform_rng(0.5, 2.0, rng);
    }

    for (int i=0; i < numGroups; i++)
        weights[i] = stan::math::dirichlet_rng(
            Eigen::VectorXd::Ones(numComponents), rng);

    for (int i=0; i < numGroups; i++) {
        for (int j=0; j < numComponents; j++)
            cluster_allocs[i][j] = j;

        for (int j=numComponents; j<samplesPerGroup[i]; j++)
            cluster_allocs[i][j] = stan::math::categorical_rng(weights[i], rng) - 1;
    }
}

void SpatialMixtureSampler::sampleAtoms()  {
    std::vector<std::vector<double>> datavec(numComponents); // one vector per component
    for (int h=0; h < numComponents; h++)
        datavec[h].reserve(numdata);

    for (int i=0; i < numGroups; i++) {
        for (int j=0; j < samplesPerGroup[i]; j++) {
            int comp = cluster_allocs[i][j];
            datavec[comp].push_back(data[i][j]);
        }
    }

    for (int h=0; h < numComponents; h++) {
        std::vector<double> params = _normalGammaUpdate(datavec[h]);
        double tau = stan::math::gamma_rng(params[1], params[2], rng);
        double sigmaNorm = 1.0 / std::sqrt(tau * params[3]);
        double mu = stan::math::normal_rng(params[0], sigmaNorm, rng);
        means[h] = mu;
        stddevs[h] = 1.0 / std::sqrt(tau);
    }
}

void SpatialMixtureSampler::sampleAllocations() {
    for (int i=0; i < numGroups; i++) {
        for (int j=0; j < samplesPerGroup[i]; j++) {
            double datum = data[i][j];
            Eigen::VectorXd logProbas(numComponents);
            for (int h=0; h < numComponents; h++) {
                logProbas(h) = \
                    std::log(weights[i][h]) + stan::math::normal_lpdf(
                        datum, means[h], stddevs[h]);
            }
            cluster_allocs[i][j] = stan::math::categorical_logit_rng(
                logProbas, rng) - 1;
        }
    }
}

void SpatialMixtureSampler::saveState(Collector<UnivariateState>* collector) {
    // First transform the state into a proto
    UnivariateState state;
    state.set_num_components(numComponents);
    for (int i=0; i < numGroups; i++) {
        UnivariateState::GroupParams* p;
        p = state.add_groupparams();
        *p->mutable_weights() = {
            weights[i].data(), weights[i].data() + numComponents};
        *p->mutable_cluster_allocs() = {
            cluster_allocs[i].begin(), cluster_allocs[i].end()};
    }
    for (int h=0; h < numComponents; h++) {
        UnivariateMixtureAtom* atom;
        atom = state.add_atoms();
        atom->set_mean(means[h]);
        atom->set_stdev(stddevs[h]);
    }
    state.set_rho(rho);
    state.mutable_sigma()->set_rows(Sigma.rows());
    state.mutable_sigma()->set_cols(Sigma.cols());
    *state.mutable_sigma()->mutable_data() = {
        Sigma.data(), Sigma.data() + Sigma.size()};

    // Then save it
    collector->collect(state);
}

std::vector<double> SpatialMixtureSampler::_normalGammaUpdate(std::vector<double> data) {
  double postMean, postA, postB, postLambda;
  int n = data.size();
  if (n == 0) {
    return std::vector<double>{priorMean, priorA, priorB, priorLambda};
  }

  double sum = std::accumulate(std::begin(data), std::end(data), 0.0);
  double ybar = sum / n;
  postMean = (priorLambda * priorMean + sum) / (priorLambda + n);
  postA = 1.0 * priorA + 1.0 * n / 2;

  double ss = 0.0;
  std::for_each(data.begin(), data.end(), [&ss, &ybar](double x) {
    ss += (x - ybar) * (x - ybar);});

  postB = (
      priorB + 0.5 * ss +
      0.5 * priorLambda / (n + priorLambda) * n *(ybar - priorMean) * (ybar - priorMean));

  postLambda = priorLambda + n;

  return std::vector<double>{postMean, postA, postB, postLambda};
}

void SpatialMixtureSampler::printDebugString() {
    std::cout << "***** Debug String ****" << std::endl;
    std::cout << "numGroups: " << numGroups <<
                 ", samplesPerGroup: ";
    for (int n: samplesPerGroup)
        std::cout << n << ", ";
    std::cout << std::endl;

    std::vector<std::vector<double>> datavec(numComponents); // one vector per component
    for (int h=0; h < numComponents; h++)
        datavec[h].reserve(numdata);

    for (int i=0; i < numGroups; i++) {
        for (int j=0; j < samplesPerGroup[i]; j++) {
            int comp = cluster_allocs[i][j];
            datavec[comp].push_back(data[i][j]);
        }
    }

    for (int h=0; h < numComponents; h++) {
        std::cout << "### Component #" << h << std::endl;
        std::cout << "##### Atom: mean=" << means[h] << ", sd=" << stddevs[h] << std::endl;
        std::cout << "###### Data: ";
        for (double d: datavec[h])
            std::cout << d << ", ";
        std::cout << std::endl;
    }
}
