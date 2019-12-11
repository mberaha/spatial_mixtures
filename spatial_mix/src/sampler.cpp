#include "sampler.hpp"
#include <numeric>

using namespace stan::math;

SpatialMixtureSampler::SpatialMixtureSampler(
        const std::vector<std::vector<double>> &_data,
        const Eigen::MatrixXd &W): data(_data), W(W) {
    numGroups = data.size();
    samplesPerGroup.resize(numGroups);
    for (int i=0; i < numGroups; i++) {
        samplesPerGroup[i] = data[i].size();
    }
    numdata = std::accumulate(
        samplesPerGroup.begin(), samplesPerGroup.end(), 0);
    pg_rng = new PolyaGammaHybridDouble(seed);
}

void SpatialMixtureSampler::init() {
    // TODO now we fix this, remember to put priors or pass from
    // constructor
    priorMean = 0.0;
    priorA = 2.0;
    priorB = 2.0;
    priorLambda = 0.1;
    rho = 0.99;
    numComponents = 10;
    nu = numComponents + 3;
    V0 = Eigen::MatrixXd::Identity(numComponents - 1, numComponents - 1);


    // Sigma = Eigen::MatrixXd::Identity(numComponents - 1, numComponents - 1);
    Sigma = inv_wishart_rng(nu, V0, rng);

    // Now proper initialization
    means.resize(numComponents);
    stddevs.resize(numComponents);
    weights = Eigen::MatrixXd::Zero(numGroups, numComponents);
    transformed_weights = Eigen::MatrixXd::Zero(numGroups, numComponents - 1);
    cluster_allocs.resize(numGroups);
    for (int i=0; i < numGroups; i++) {
        cluster_allocs[i].resize(samplesPerGroup[i]);
    }

    for (int h=0; h < numComponents; h++) {
        means[h] = normal_rng(0.0, 10.0, rng);
        stddevs[h] = uniform_rng(0.5, 2.0, rng);
    }

    for (int i=0; i < numGroups; i++)
        weights.row(i) = dirichlet_rng(
            Eigen::VectorXd::Ones(numComponents), rng);

    for (int i=0; i < numGroups; i++) {
        for (int j=0; j < numComponents; j++)
            cluster_allocs[i][j] = j;

        for (int j=numComponents; j<samplesPerGroup[i]; j++)
            cluster_allocs[i][j] = categorical_rng(weights.row(i), rng) - 1;
    }

    // normalize W
    for (int i=0; i<W.rows(); ++i){
      W.row(i) *= rho/W.row(i).sum();
    }

    // compute inverses of sigma(-h,h))
    pippo.resize(numComponents - 1);
    sigma_star_h.resize(numComponents - 1);

    _computeInvSigmaH();
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
                    std::log(weights(i, h)) + normal_lpdf(
                        datum, means[h], stddevs[h]);
            }
            cluster_allocs[i][j] = categorical_logit_rng(
                logProbas, rng) - 1;
        }
    }
}

void SpatialMixtureSampler::sampleWeights() {

    for (int i=0; i < numGroups; i++) {
        for (int h=0; h < numComponents; h++) {
            /* we draw omega from a Polya-Gamma distribution
               TODO The second parameter can be computed from the weights
            */
            // TODO log sum exp
            double C_ih = log(exp(transformed_weights.row(i)).sum()
                - exp(transformed_weights(i, h)));
            double omega_ih = pg_rng->draw(
                samplesPerGroup[i],
                transformed_weights(i, h) - C_ih
                );

            Eigen::VectorXd mu_i = W.row(i) * transformed_weights;
            double mu_star_ih = mu_i[h] + pippo[h].dot(
                utils::removeColumn(transformed_weights, h).row(i) -
                utils::removeRow(mu_i, h));
            double sigma_hat_ih = (1.0/sigma_star_h[h] + omega_ih);
            int N_ih = std::count(cluster_allocs[i].begin(),
                                  cluster_allocs[i].end(), h);
            double mu_hat_ih = (mu_star_ih/sigma_star_h[h] + N_ih -
                0.5 * samplesPerGroup[i] + omega_ih*C_ih)/(sigma_hat_ih);

            transformed_weights(i, h) =
                normal_rng(mu_hat_ih, std::sqrt(sigma_hat_ih), rng);

        }
        weights.row(i) = utils::InvAlr(transformed_weights.row(i));
    }
}

void SpatialMixtureSampler::sampleSigma() {
    Eigen::MatrixXd Vn = V0;
    double nu_n = nu + numGroups;

    for (int i=0; i < numGroups; i++) {
        Eigen::VectorXd mu_i = W.row(i) * transformed_weights;
        Vn += (transformed_weights.row(i).transpose() - mu_i) *
              (transformed_weights.row(i).transpose() - mu_i).transpose();
    }

    Sigma = inv_wishart_rng(nu_n, Vn, rng);
    _computeInvSigmaH();
}

void SpatialMixtureSampler::_computeInvSigmaH() {
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(
        numComponents - 2, numComponents - 2);

    for(int h=0; h<numComponents-1;++h){
        pippo[h] = utils::removeColumn(Sigma, h).row(h) *
                   utils::removeRowColumn(Sigma, h).llt().solve(I);
    }

    for(int h=0; h<sigma_star_h.size();++h){
        double aux = pippo[h].dot(utils::removeRow(Sigma, h).col(h));
        sigma_star_h[h] = Sigma(h, h) - aux;
    }
}

void SpatialMixtureSampler::saveState(Collector<UnivariateState>* collector) {
    collector->collect(getStateAsProto());
}

UnivariateState SpatialMixtureSampler::getStateAsProto() {
    UnivariateState state;
    state.set_num_components(numComponents);
    for (int i=0; i < numGroups; i++) {
        UnivariateState::GroupParams* p;
        p = state.add_groupparams();
        *p->mutable_weights() = {
            weights.row(i).data(), weights.row(i).data() + numComponents};
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
    return state;
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

    std::cout << "Sigma: \n" << Sigma << std::endl << std::endl;

    for (int h=0; h < numComponents; h++) {
        std::cout << "### Component #" << h << std::endl;
        std::cout << "##### Atom: mean=" << means[h] << ", sd=" << stddevs[h]
                  << " weights per group: " << weights.col(h).transpose() << std::endl;
        // std::cout << "###### Data: ";
        // for (double d: datavec[h])
        //     std::cout << d << ", ";
        std::cout << std::endl;
    }
}
