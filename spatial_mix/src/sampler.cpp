#include "sampler.hpp"

using namespace stan::math;

SpatialMixtureSampler::SpatialMixtureSampler(
        const SamplerParams &_params,
        const std::vector<std::vector<double>> &_data,
        const Eigen::MatrixXd &W): params(_params), data(_data), W_init(W) {
    numGroups = data.size();
    samplesPerGroup.resize(numGroups);
    for (int i=0; i < numGroups; i++) {
        samplesPerGroup[i] = data[i].size();
    }
    numdata = std::accumulate(
        samplesPerGroup.begin(), samplesPerGroup.end(), 0);
}


SpatialMixtureSampler::SpatialMixtureSampler(
        const SamplerParams &_params,
        const std::vector<std::vector<double>> &_data,
        const Eigen::MatrixXd &W, const std::vector<Eigen::MatrixXd> &X):
            params(_params), data(_data), W_init(W) {
    numGroups = data.size();
    samplesPerGroup.resize(numGroups);
    for (int i=0; i < numGroups; i++) {
        samplesPerGroup[i] = data[i].size();
    }
    numdata = std::accumulate(
        samplesPerGroup.begin(), samplesPerGroup.end(), 0);

    if (X.size() > 0) {
        regression = true;
        p_size = X[0].cols();
        std::cout << "p_size: " << p_size << std::endl;
        reg_coeff_mean = Eigen::VectorXd::Zero(p_size);
        reg_coeff_prec = Eigen::MatrixXd::Identity(p_size, p_size);
        reg_coeff = stan::math::multi_normal_rng(
            reg_coeff_mean, 10 * reg_coeff_prec, rng);

        predictors.resize(numdata, p_size);
        reg_data.resize(numdata);
        V.resize(numdata);
        mu.resize(numdata);
        int start = 0;
        std::cout << "H" << std::endl;
        std::cout << "numGroups: " << numGroups << std::endl;
        for (int i=0; i < numGroups; i++) {
            std::cout << "i: " << i << ", start: " << start << ", samples: "
                      << samplesPerGroup[i] << std::endl;
            predictors.block(start, 0, samplesPerGroup[i], p_size) = X[i];
            std::cout << "predictors ok" << std::endl;
            reg_data.segment(start, samplesPerGroup[i]) = \
                Eigen::Map<Eigen::VectorXd>(data[i].data(), samplesPerGroup[i]);
            std::cout << "reg data ok" << std::endl;
            start += samplesPerGroup[i];
        }
        computeRegressionResiduals();
    }
}

void SpatialMixtureSampler::init() {
    pg_rng = new PolyaGammaHybridDouble(seed);

    numComponents = params.num_components();

    priorMean = params.p0_params().mu0();
    priorA = params.p0_params().a();
    priorB = params.p0_params().b();
    priorLambda = params.p0_params().lam_();

    nu = params.sigma_params().nu();
    if (params.sigma_params().identity())
        V0 = Eigen::MatrixXd::Identity(numComponents - 1, numComponents - 1);
    else
        throw std::logic_error("Case not implemented yet");

    alpha = params.rho_params().a();
    beta = params.rho_params().b();

    // Now proper initialization
    rho = 0.5;
    Sigma = Eigen::MatrixXd::Identity(numComponents - 1, numComponents - 1);
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
    std::cout << "init done" << std::endl;
}

void SpatialMixtureSampler::sample()  {
    if (regression) {
        regress();
        computeRegressionResiduals();
    }
    sampleAtoms();
    sampleAllocations();
    sampleWeights();
    sampleSigma();
    sampleRho();
}


void SpatialMixtureSampler::sampleAtoms()  {
    std::vector<std::vector<double>> datavec(numComponents);
    for (int h=0; h < numComponents; h++)
        datavec[h].reserve(numdata);

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
    for (int i=0; i < numGroups; i++) {
        #pragma omp parallel for
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

        #pragma omp parallel for
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

    #pragma omp parallel for
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

void SpatialMixtureSampler::regress() {
    // Compute mu and v
    int start = 0;
    int s = 0;

    for (int i=0; i < numGroups; i++) {
        #pragma omp parallel for
        for (int j=0; j < samplesPerGroup[i]; j++) {
            s = cluster_allocs[i][j];
            mu(start + j) = means[s];
            V.diagonal()[start + j] = 1.0 / (stddevs[s] * stddevs[s]);
        }
        start += samplesPerGroup[i];
    }

    // compute posterior parameters for beta
    Eigen::VectorXd postMean(p_size);
    Eigen::MatrixXd postPrec(p_size, p_size);

    postPrec = predictors.transpose() * V * predictors + reg_coeff_prec;
    postMean = postPrec.ldlt().solve(
        predictors.transpose() * V * (reg_data - mu));

    reg_coeff = stan::math::multi_normal_prec_rng(postMean, postPrec, rng);
}

void SpatialMixtureSampler::computeRegressionResiduals() {
    Eigen::VectorXd residuals = reg_data - predictors * reg_coeff;
    int start = 0;

    for (int i=0; i < numGroups; i++) {
        #pragma omp parallel for
        for (int j=0; j < samplesPerGroup[i]; j++) {
            data[i][j] = residuals[start + j];
        }
        start += samplesPerGroup[i];
    }
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

    if (regression)
        *state.mutable_regression_coefficients() = {
            reg_coeff.data(), reg_coeff.data() + p_size};
    return state;
}

void SpatialMixtureSampler::printDebugString() {
    std::cout << "***** Debug String ****" << std::endl;
    std::cout << "numGroups: " << numGroups <<
                 ", samplesPerGroup: ";
    for (int n: samplesPerGroup)
        std::cout << n << ", ";
    std::cout << std::endl;

    // one vector per component
    std::vector<std::vector<std::vector<double>>> datavecs(numGroups);

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

    if (regression) {
        std::cout << "Regression Coefficients: " << std::endl;
        std::cout << "    " << reg_coeff.transpose() << std::endl;
    }
}
