#include "sampler.hpp"
#include <numeric>

using namespace stan::math;

SpatialMixtureSampler::SpatialMixtureSampler(
        const std::vector<std::vector<double>> &_data,
        const Eigen::MatrixXd &W): data(_data), W_init(W) {
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
    rho = 0.9;
    numComponents = 3;
    nu = numComponents + 2;
    V0 = Eigen::MatrixXd::Identity(numComponents - 1, numComponents - 1);
    alpha = 5;
    beta = 5;


    Sigma = Eigen::MatrixXd::Identity(numComponents - 1, numComponents - 1);
    // Sigma = inv_wishart_rng(nu, V0, rng);

    // Now proper initialization
    means.resize(numComponents);
    stddevs.resize(numComponents);
    weights = Eigen::MatrixXd::Zero(numGroups, numComponents);
    transformed_weights = Eigen::MatrixXd::Zero(numGroups, numComponents);
    cluster_allocs.resize(numGroups);
    for (int i=0; i < numGroups; i++) {
        cluster_allocs[i].resize(samplesPerGroup[i]);
    }

    for (int h=0; h < numComponents; h++) {
        means[h] = normal_rng(0.0, 10.0, rng);
        stddevs[h] = uniform_rng(0.5, 2.0, rng);
    }

    for (int i=0; i < numGroups; i++) {
        weights.row(i) = dirichlet_rng(
            Eigen::VectorXd::Ones(numComponents), rng);
        transformed_weights.row(i) = utils::Alr(weights.row(i), true);
    }


    for(int i=0; i< numGroups; ++i){
      std::cout << "GROUP: " << i+1 << std::endl;
      for (int h=0; h<numComponents-1; h++){
          std::cout << "transformed_weight: " << transformed_weights(i, h) << std::endl;
      }
    }

    for (int i=0; i < numGroups; i++) {
        for (int j=0; j < numComponents; j++)
            cluster_allocs[i][j] = j;

        for (int j=numComponents; j<samplesPerGroup[i]; j++)
            cluster_allocs[i][j] = categorical_rng(weights.row(i), rng) - 1;
    }

    W = W_init;
    // normalize W
    for (int i=0; i < W.rows(); ++i){
      W.row(i) *= rho/W.row(i).sum();
    }

    F = Eigen::MatrixXd::Zero(numGroups, numGroups);
    for (int i=0; i < numGroups; i++) {
        F(i, i) = W_init.row(i).sum();
    }

    // compute inverses of sigma(-h,h))
    pippo.resize(numComponents - 1);
    sigma_star_h.resize(numComponents - 1);

    _computeInvSigmaH();
    std::cout<<"init done nostro"<<std::endl;
}

void SpatialMixtureSampler::sample()  {
    sampleAtoms();
    sampleAllocations();
    sampleWeights();
    sampleSigma();
    sampleRho();
}


void SpatialMixtureSampler::sampleAtoms()  {
    std::vector<std::vector<double>> datavec(numComponents); // one vector per component
    for (int h=0; h < numComponents; h++)
        datavec[h].reserve(numdata);

    #pragma omp parallel for
    for (int i=0; i < numGroups; i++) {
        for (int j=0; j < samplesPerGroup[i]; j++) {
            int comp = cluster_allocs[i][j];
            datavec[comp].push_back(data[i][j]);
        }
    }

    #pragma omp parallel for
    for (int h=0; h < numComponents; h++) {
        std::vector<double> params = utils::normalGammaUpdate(
            datavec[h], priorMean, priorA, priorB, priorLambda);
        double tau = stan::math::gamma_rng(params[1], params[2], rng);
        double sigmaNorm = 1.0 / std::sqrt(tau * params[3]);
        double mu = stan::math::normal_rng(params[0], sigmaNorm, rng);
        means[h] = mu;
        stddevs[h] = 1.0 / std::sqrt(tau);
    }
}

void SpatialMixtureSampler::sampleAllocations() {
    #pragma omp parallel for
    for (int i=0; i < numGroups; i++) {
        for (int j=0; j < samplesPerGroup[i]; j++) {
            double datum = data[i][j];
            Eigen::VectorXd logProbas(numComponents);
            for (int h=0; h < numComponents; h++) {
                logProbas(h) = \
                    std::log(weights(i, h) + 1e-6) + normal_lpdf(
                        datum, means[h], stddevs[h]);
            }
            Eigen::VectorXd probas = logProbas.array().exp();
            probas /= probas.sum();
            cluster_allocs[i][j] = categorical_rng(
                probas, rng) - 1;
        }
    }
}

void SpatialMixtureSampler::sampleWeights() {
    for (int i=0; i < numGroups; i++) {
        std::vector<int> cluster_sizes(numComponents, 0);
        for(int j=0; j<samplesPerGroup[i]; j++)
            cluster_sizes[cluster_allocs[i][j]] += 1;
        for (int h=0; h < numComponents - 1; h++) {
            /*
             * we draw omega from a Polya-Gamma distribution
             */
            Eigen::VectorXd weightsForCih = utils::removeElem(
                transformed_weights.row(i), h);
            double C_ih = stan::math::log_sum_exp(weightsForCih);

            double omega_ih = pg_rng->draw(
                samplesPerGroup[i],
                transformed_weights(i, h) - C_ih);

            Eigen::VectorXd mu_i = W.row(i) * transformed_weights;
            mu_i = mu_i.head(numComponents - 1);
            Eigen::VectorXd wtilde = transformed_weights.row(i).head(numComponents - 1);

            double mu_star_ih = mu_i[h] + pippo[h].dot(
                utils::removeElem(wtilde , h) -
                utils::removeElem(mu_i, h));

            double sigma_hat_ih = 1.0 / (1.0 / sigma_star_h[h] + omega_ih);
            int N_ih = cluster_sizes[h];
            double mu_hat_ih = (
                mu_star_ih / sigma_star_h[h] + N_ih -
                0.5 * samplesPerGroup[i] + omega_ih*C_ih) * (sigma_hat_ih);

            transformed_weights(i, h) = normal_rng(
                mu_hat_ih, std::sqrt(sigma_hat_ih), rng);

        }
        weights.row(i) = utils::InvAlr(transformed_weights.row(i), true);

    }
    for (int i=0; i < numGroups; i++)
        transformed_weights.row(i) = utils::Alr(weights.row(i), true);
}


// We use a MH step with a truncated normal proposal
void SpatialMixtureSampler::sampleRho() {
    double curr = rho;
    double sigma = 0.1;
    double proposed = utils::trunc_normal_rng(curr, sigma, 0.0, 1.0, rng);

    // compute acceptance ratio
    Eigen::MatrixXd rowVar = F - proposed * W_init;
    Eigen::MatrixXd meanMat = Eigen::MatrixXd::Zero(numGroups, numComponents - 1);
    double num = stan::math::beta_lpdf(proposed, alpha, beta) +
                 stan::math::matrix_normal_prec_lpdf(
                    utils::removeColumn(transformed_weights, numComponents -1),
                    meanMat, rowVar, SigmaInv) +
                 utils::trunc_normal_lpdf(proposed, curr, sigma, 0.0, 1.0);

    rowVar = F - curr * W_init;
    double den = stan::math::beta_lpdf(curr, alpha, beta) +
                 stan::math::matrix_normal_prec_lpdf(
                    utils::removeColumn(transformed_weights, numComponents -1),
                    meanMat, rowVar, SigmaInv) +
                 utils::trunc_normal_lpdf(curr, proposed, sigma, 0.0, 1.0);

    double arate = std::min(1.0, std::exp(num-den));
    if (stan::math::uniform_rng(0.0, 1.0, rng) < arate) {
        rho = proposed;
        numAccepted += 1;
        W = W_init;
        // normalize W
        for (int i=0; i < W.rows(); ++i){
          W.row(i) *= rho/W.row(i).sum();
        }
    }
}

void SpatialMixtureSampler::sampleSigma() {
    Eigen::MatrixXd Vn = V0;
    double nu_n = nu + numGroups;

    #pragma omp parallel for
    for (int i=0; i < numGroups; i++) {
        Eigen::VectorXd mu_i = W.row(i) * utils::removeColumn(
            transformed_weights, numComponents-1);
        Eigen::VectorXd wtilde_i = utils::removeElem(
            transformed_weights.row(i), numComponents-1);
        Vn += (wtilde_i - mu_i) * (wtilde_i - mu_i).transpose();
    }

    Sigma = inv_wishart_rng(nu_n, Vn, rng);
    _computeInvSigmaH();
}

void SpatialMixtureSampler::_computeInvSigmaH() {
    SigmaInv = Sigma.llt().solve(Eigen::MatrixXd::Identity(
        numComponents - 1, numComponents - 1));

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(
        numComponents - 2, numComponents - 2);

    #pragma omp parallel for
    for(int h=0; h < numComponents - 1; ++h){
        pippo[h] = utils::removeColumn(Sigma, h).row(h) *
                   utils::removeRowColumn(Sigma, h).llt().solve(I);
    }

    #pragma omp parallel for
    for(int h=0; h < sigma_star_h.size(); ++h){
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
        Eigen::VectorXd w = weights.row(i);

        *p->mutable_weights() = {w.data(),w.data() + numComponents};
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

void SpatialMixtureSampler::printDebugString() {
    std::cout << "***** Debug String ****" << std::endl;
    std::cout << "numGroups: " << numGroups <<
                 ", samplesPerGroup: ";
    for (int n: samplesPerGroup)
        std::cout << n << ", ";
    std::cout << std::endl;

    std::vector<std::vector<std::vector<double>>> datavecs(numGroups); // one vector per component

    for (int i=0; i < numGroups; i++) {
        datavecs[i].resize(numComponents);
        for (int j=0; j < samplesPerGroup[i]; j++) {
            int comp = cluster_allocs[i][j];
            datavecs[i][comp].push_back(data[i][j]);
        }
    }

    std::cout << "Sigma: \n" << Sigma << std::endl << std::endl;

    for (int h=0; h < numComponents; h++) {
        std::cout << "### Component #" << h << std::endl;
        std::cout << "##### Atom: mean=" << means[h] << ", sd=" << stddevs[h]
                  << " weights per group: " << weights.col(h).transpose() << std::endl;
        std::cout << std::endl;
    }
}
