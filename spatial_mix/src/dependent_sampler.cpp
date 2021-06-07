#include "dependent_sampler.hpp"

using namespace stan::math;

DependentSpatialMixtureSampler::DependentSpatialMixtureSampler(
    const SamplerParams &_params, const std::vector<std::vector<double>> &_data,
    const Eigen::MatrixXd &W, const std::vector<Eigen::MatrixXd> &X)
    : params(_params), data(_data), W_init(W), predictors(X) {
  numGroups = data.size();
  samplesPerGroup.resize(numGroups);
  for (int i = 0; i < numGroups; i++) {
    samplesPerGroup[i] = data[i].size();
  }
  numdata = std::accumulate(samplesPerGroup.begin(), samplesPerGroup.end(), 0);

  p_size = X[0].cols();
}

void DependentSpatialMixtureSampler::init() {
  pg_rng = new PolyaGammaHybridDouble(seed);

  numComponents = params.num_components();
  mtilde_sigmasq = params.mtilde_sigmasq();

  beta0 = Eigen::VectorXd::Ones(p_size) * params.linreg_params().mean();
  beta_prec =
      Eigen::MatrixXd::Identity(p_size, p_size) * params.linreg_params().prec();

  priorA = params.linreg_params().a();
  priorB = params.linreg_params().b();

  nu = params.sigma_params().nu();
  if (params.sigma_params().identity())
    V0 = Eigen::MatrixXd::Identity(numComponents - 1, numComponents - 1);
  else {
    V0 = Eigen::MatrixXd::Identity(numComponents - 1, numComponents - 1);
    std::cout << "Case not implemented yet, settig V0 to identity" << std::endl;
  }

  alpha = params.rho_params().a();
  beta = params.rho_params().b();

  node2comp = utils::findConnectedComponents(W_init);
  auto it = std::max_element(node2comp.begin(), node2comp.end());
  num_connected_comps = *it + 1;

  comp2node.resize(num_connected_comps);
  for (int i = 0; i < numGroups; i++) {
    comp2node[node2comp[i]].push_back(i);
  }

  // Now proper initialization
  rho = 0.99;
  rho_sum = 0;
  rho_sum_sq = 0;
  Sigma = Eigen::MatrixXd::Identity(numComponents - 1, numComponents - 1);
  betas.resize(numComponents);
  stddevs.resize(numComponents);
  weights = Eigen::MatrixXd::Zero(numGroups, numComponents);
  transformed_weights = Eigen::MatrixXd::Zero(numGroups, numComponents);
  cluster_allocs.resize(numGroups);
  for (int i = 0; i < numGroups; i++) {
    cluster_allocs[i].resize(samplesPerGroup[i]);
  }

  for (int h = 0; h < numComponents; h++) {
    betas[h] = multi_normal_prec_rng(beta0, beta_prec * 0.1, rng);
    stddevs[h] = uniform_rng(0.5, 2.0, rng);
  }

  for (int i = 0; i < numGroups; i++) {
    weights.row(i) = dirichlet_rng(Eigen::VectorXd::Ones(numComponents), rng);
    transformed_weights.row(i) = utils::Alr(weights.row(i), true);
  }

  for (int i = 0; i < numGroups; i++) {
    for (int j = 0; j < std::min(numComponents, samplesPerGroup[i]); j++)
      cluster_allocs[i][j] = j;

    for (int j = numComponents; j < samplesPerGroup[i]; j++)
      cluster_allocs[i][j] = categorical_rng(weights.row(i), rng) - 1;
  }

  // W = W_init;
  // // normalize W
  // for (int i=0; i < W.rows(); ++i){
  //   W.row(i) *= rho/W.row(i).sum();
  // }

  F = Eigen::MatrixXd::Zero(numGroups, numGroups);
  for (int i = 0; i < numGroups; i++)
    F(i, i) = rho * W_init.row(i).sum() + (1 - rho);

  F_by_comp.resize(num_connected_comps);
  G_by_comp.resize(num_connected_comps);
  for (int k = 0; k < num_connected_comps; k++) {
    Eigen::MatrixXd curr_f =
        Eigen::MatrixXd::Zero(comp2node[k].size(), comp2node[k].size());
    Eigen::MatrixXd curr_g =
        Eigen::MatrixXd::Zero(comp2node[k].size(), comp2node[k].size());
    for (int i = 0; i < comp2node[k].size(); i++) {
      curr_f(i, i) = F(comp2node[k][i], comp2node[k][i]);
      for (int j = 0; j < comp2node[k].size(); j++) {
        curr_g(i, j) = W_init(comp2node[k][i], comp2node[k][j]);
      }
    }
    F_by_comp[k] = curr_f;
    G_by_comp[k] = curr_g;
  }

  // last component is not used!
  mtildes = Eigen::MatrixXd::Zero(num_connected_comps, numComponents);

  pippo.resize(numComponents - 1);
  sigma_star_h = Eigen::MatrixXd::Zero(numGroups, numComponents - 1);

  _computeInvSigmaH();

  std::cout << "init done" << std::endl;
}

void DependentSpatialMixtureSampler::sample() {
  sampleAtoms();
  sampleAllocations();
  sampleWeights();
  sampleSigma();
  // sampleRho();
  sample_mtilde();
}

void DependentSpatialMixtureSampler::sampleAtoms() {
  std::vector<std::vector<double>> datavec(numComponents);
  std::vector<Eigen::MatrixXd> covariatevec(numComponents);

  for (int h = 0; h < numComponents; h++) {
    datavec[h].reserve(numdata);
  }

  for (int i = 0; i < numGroups; i++) {
    for (int j = 0; j < samplesPerGroup[i]; j++) {
      int comp = cluster_allocs[i][j];
      datavec[comp].push_back(data[i][j]);
      utils::append_by_row(&covariatevec[comp], predictors[i].row(j));
    }
  }

  std::vector<Eigen::VectorXd> y_by_comp(numComponents);
  for (int h = 0; h < numComponents; h++) {
    Eigen::Map<Eigen::VectorXd> curr(datavec[h].data(), datavec[h].size());
    y_by_comp[h] = curr;
  }

  // #pragma omp parallel for
  for (int h = 0; h < numComponents; h++) {
    double sigma;
    Eigen::VectorXd beta;
    if (y_by_comp[h].size() == 0) {
      sigma = std::sqrt(stan::math::inv_gamma_rng(priorA, priorB, rng));
      beta = stan::math::multi_normal_prec_rng(beta0, beta_prec, rng);
    } else {
      const Eigen::MatrixXd &X = covariatevec[h];
      const Eigen::VectorXd &y = y_by_comp[h];
      const double &curr_sigma = stddevs[h];

      Eigen::MatrixXd xtransx = X.transpose() * X;
      assert(curr_sigma > 0.01);
      assert(xtransx.rows() == beta_prec.rows());
      assert(xtransx.cols() == beta_prec.cols());

      Eigen::MatrixXd prec_beta_post =
          xtransx / (curr_sigma * curr_sigma) + beta_prec;

      Eigen::VectorXd mean_beta_post = prec_beta_post.ldlt().solve(
          X.transpose() * y / (curr_sigma * curr_sigma) + beta_prec * beta0);

      beta = stan::math::multi_normal_prec_rng(mean_beta_post, prec_beta_post,
                                               rng);

      double a = priorA + y.size() / 2.0;
      double b = priorB + (y - X * beta).squaredNorm();
      sigma = std::sqrt(stan::math::inv_gamma_rng(a, b, rng));
    }
    betas[h] = beta;
    stddevs[h] = sigma;
  }
}

void DependentSpatialMixtureSampler::sampleAllocations() {
  for (int i = 0; i < numGroups; i++) {
    // #pragma omp parallel for
    for (int j = 0; j < samplesPerGroup[i]; j++) {
      double datum = data[i][j];
      Eigen::VectorXd cov = predictors[i].row(j);
      Eigen::VectorXd logProbas(numComponents);
      for (int h = 0; h < numComponents; h++) {
        logProbas(h) = std::log(weights(i, h) + 1e-6) +
                       normal_lpdf(datum, betas[h].dot(cov), stddevs[h]);
      }
      Eigen::VectorXd probas = logProbas.array().exp();
      probas /= probas.sum();
      cluster_allocs[i][j] = categorical_rng(probas, rng) - 1;
    }
  }
}

void DependentSpatialMixtureSampler::sampleWeights() {
  for (int i = 0; i < numGroups; i++) {
    std::vector<int> cluster_sizes(numComponents, 0);

    // #pragma omp parallel for
    for (int j = 0; j < samplesPerGroup[i]; j++)
      cluster_sizes[cluster_allocs[i][j]] += 1;

    for (int h = 0; h < numComponents - 1; h++) {
      /*
       * we draw omega from a Polya-Gamma distribution
       */
      Eigen::VectorXd weightsForCih =
          utils::removeElem(transformed_weights.row(i), h);
      double C_ih = stan::math::log_sum_exp(weightsForCih);

      double omega_ih =
          pg_rng->draw(samplesPerGroup[i], transformed_weights(i, h) - C_ih);

      Eigen::VectorXd mu_i =
          (W_init.row(i) * transformed_weights).array() * rho +
          mtildes.row(node2comp[i]).array() * (1 - rho);
      mu_i = mu_i.array() / (W_init.row(i).sum() * rho + 1 - rho);
      mu_i = mu_i.head(numComponents - 1);
      Eigen::VectorXd wtilde =
          transformed_weights.row(i).head(numComponents - 1);

      double mu_star_ih = mu_i[h] + pippo[h].dot(utils::removeElem(wtilde, h) -
                                                 utils::removeElem(mu_i, h));

      double sigma_hat_ih = 1.0 / (1.0 / sigma_star_h(i, h) + omega_ih);
      int N_ih = cluster_sizes[h];
      double mu_hat_ih = (mu_star_ih / sigma_star_h(i, h) + N_ih -
                          0.5 * samplesPerGroup[i] + omega_ih * C_ih) *
                         (sigma_hat_ih);

      transformed_weights(i, h) =
          normal_rng(mu_hat_ih, std::sqrt(sigma_hat_ih), rng);
    }
    weights.row(i) = utils::InvAlr(transformed_weights.row(i), true);
  }

  // #pragma omp parallel for
  for (int i = 0; i < numGroups; i++)
    transformed_weights.row(i) = utils::Alr(weights.row(i), true);
}

// We use a MH step with a truncated normal proposal
void DependentSpatialMixtureSampler::sampleRho() {
  iter += 1;
  double curr = rho;
  double sigma;
  if (iter < 3) {
    sigma = 0.01;
  } else {
    if (sigma_n_rho == 0)
      sigma = 0.01;
    else {
      if (stan::math::uniform_rng(0.0, 1.0, rng) < 0.05)
        sigma = 0.01;
      else
        sigma = 2.38 * sigma_n_rho;
    }
  }
  double proposed = utils::trunc_normal_rng(curr, sigma, 0.0, 0.9999, rng);

  // compute acceptance ratio
  Eigen::MatrixXd row_prec = F - proposed * W_init;

  Eigen::MatrixXd temp =
      utils::removeColumn(transformed_weights, numComponents - 1);
  Eigen::MatrixXd meanmat = Eigen::MatrixXd::Zero(numGroups, numComponents - 1);
  for (int i = 0; i < numGroups; i++)
    meanmat.row(i) = mtildes.row(node2comp[i]).head(numComponents - 1);

  double num =
      stan::math::beta_lpdf(proposed, alpha, beta) +
      utils::matrix_normal_prec_lpdf(temp, meanmat, row_prec, SigmaInv) +
      utils::trunc_normal_lpdf(proposed, curr, sigma, 0.0, 1);

  row_prec = F - curr * W_init;
  double den =
      stan::math::beta_lpdf(curr, alpha, beta) +
      utils::matrix_normal_prec_lpdf(temp, meanmat, row_prec, SigmaInv) +
      utils::trunc_normal_lpdf(curr, proposed, sigma, 0.0, 1);

  double arate = std::min(1.0, std::exp(num - den));
  if (stan::math::uniform_rng(0.0, 1.0, rng) < arate) {
    rho = proposed;
    numAccepted += 1;

    F = Eigen::MatrixXd::Zero(numGroups, numGroups);
    for (int i = 0; i < numGroups; i++)
      F(i, i) = rho * W_init.row(i).sum() + (1 - rho);

    F_by_comp.resize(num_connected_comps);
    G_by_comp.resize(num_connected_comps);
    for (int k = 0; k < num_connected_comps; k++) {
      Eigen::MatrixXd curr_f =
          Eigen::MatrixXd::Zero(comp2node[k].size(), comp2node[k].size());
      Eigen::MatrixXd curr_g =
          Eigen::MatrixXd::Zero(comp2node[k].size(), comp2node[k].size());
      for (int i = 0; i < comp2node[k].size(); i++) {
        curr_f(i, i) = F(comp2node[k][i], comp2node[k][i]);
        for (int j = 0; j < comp2node[k].size(); j++) {
          curr_g(i, j) = W_init(comp2node[k][i], comp2node[k][j]);
        }
      }
      F_by_comp[k] = curr_f;
      G_by_comp[k] = curr_g;
    }
  }

  // update adaptive MCMC params
  rho_sum += rho;
  rho_sum_sq += rho * rho;
  double rho_mean = rho_sum / iter;
  sigma_n_rho = rho_sum_sq / iter - rho_mean * rho_mean;
}

void DependentSpatialMixtureSampler::sampleSigma() {
  Eigen::MatrixXd Vn = V0;
  double nu_n = nu + numGroups;
  Eigen::MatrixXd F_m_rhoG = F - W_init * rho;

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < numGroups; i++) {
    Eigen::VectorXd wtilde_i =
        transformed_weights.row(i).head(numComponents - 1);
    Eigen::VectorXd mtilde_i =
        mtildes.row(node2comp[i]).head(numComponents - 1);
    for (int j = 0; j < numGroups; j++) {
      Eigen::VectorXd wtilde_j =
          transformed_weights.row(j).head(numComponents - 1);
      Eigen::VectorXd mtilde_j =
          mtildes.row(node2comp[j]).head(numComponents - 1);
      Vn += ((wtilde_i - mtilde_i) * (wtilde_j - mtilde_j).transpose()) *
            F_m_rhoG(i, j);
    }
  }
  Sigma = inv_wishart_rng(nu_n, Vn, rng);
  _computeInvSigmaH();
}

void DependentSpatialMixtureSampler::_computeInvSigmaH() {
  SigmaInv = Sigma.llt().solve(
      Eigen::MatrixXd::Identity(numComponents - 1, numComponents - 1));

  Eigen::MatrixXd I =
      Eigen::MatrixXd::Identity(numComponents - 2, numComponents - 2);

  // #pragma omp parallel for
  for (int h = 0; h < numComponents - 1; ++h) {
    pippo[h] = utils::removeColumn(Sigma, h).row(h) *
               utils::removeRowColumn(Sigma, h).llt().solve(I);
  }

  // #pragma omp parallel for
  for (int h = 0; h < numComponents - 1; ++h) {
    double aux = pippo[h].dot(utils::removeRow(Sigma, h).col(h));
    for (int i = 0; i < numGroups; i++) {
      sigma_star_h(i, h) = (Sigma(h, h) - aux) / F(i, i);
    }
  }
}

void DependentSpatialMixtureSampler::sample_mtilde() {
  int H = numComponents;
  Eigen::MatrixXd prec_prior =
      Eigen::MatrixXd::Identity(numComponents - 1, numComponents - 1).array() *
      (1.0 / mtilde_sigmasq);

  Eigen::MatrixXd F_min_rhoG = F - rho * W_init;

  for (int k = 1; k < num_connected_comps; k++) {
    Eigen::VectorXd currweights(comp2node[k].size() * (numComponents - 1));
    for (int i = 0; i < comp2node[k].size(); i++) {
      currweights.segment(i * (H - 1), (H - 1)) =
          transformed_weights.row(comp2node[k][i]).head(H - 1).transpose();
    }

    Eigen::MatrixXd curr_f_min_rhoG = F_by_comp[k] - rho * G_by_comp[k];

    Eigen::MatrixXd curr_prec = kroneckerProduct(curr_f_min_rhoG, SigmaInv);

    Eigen::MatrixXd I_star =
        kroneckerProduct(Eigen::VectorXd::Ones(comp2node[k].size()),
                         Eigen::MatrixXd::Identity((H - 1), (H - 1)));

    Eigen::MatrixXd prec_post =
        I_star.transpose() * curr_prec * I_star + prec_prior;
    Eigen::VectorXd m_post =
        prec_post.ldlt().solve(I_star.transpose() * curr_prec * currweights);

    Eigen::VectorXd sampled =
        stan::math::multi_normal_prec_rng(m_post, prec_post, rng);

    mtildes.row(k).head(H - 1) = sampled;
  }
}

void DependentSpatialMixtureSampler::saveState(
    Collector<DependentState> *collector) {
  collector->collect(getStateAsProto());
}

DependentState DependentSpatialMixtureSampler::getStateAsProto() {
  DependentState state;
  state.set_num_components(numComponents);
  for (int i = 0; i < numGroups; i++) {
    DependentState::GroupParams *p;
    p = state.add_groupparams();
    Eigen::VectorXd w = weights.row(i);

    *p->mutable_weights() = {w.data(), w.data() + numComponents};
    *p->mutable_cluster_allocs() = {cluster_allocs[i].begin(),
                                    cluster_allocs[i].end()};
  }
  for (int h = 0; h < numComponents; h++) {
    DependentMixtureAtom *atom;
    atom = state.add_atoms();
    *atom->mutable_beta() = {betas[h].data(),
                             betas[h].data() + betas[h].size()};
    atom->set_stdev(stddevs[h]);
  }
  state.set_rho(rho);
  state.mutable_sigma()->set_rows(Sigma.rows());
  state.mutable_sigma()->set_cols(Sigma.cols());
  *state.mutable_sigma()->mutable_data() = {Sigma.data(),
                                            Sigma.data() + Sigma.size()};
  return state;
}

void DependentSpatialMixtureSampler::printDebugString() {
  std::cout << "***** Debug String ****" << std::endl;
  std::cout << "numGroups: " << numGroups << ", samplesPerGroup: ";
  for (int n : samplesPerGroup) std::cout << n << ", ";
  std::cout << std::endl;

  // one vector per component
  std::vector<std::vector<std::vector<double>>> datavecs(numGroups);

  for (int i = 0; i < numGroups; i++) {
    datavecs[i].resize(numComponents);
    for (int j = 0; j < samplesPerGroup[i]; j++) {
      int comp = cluster_allocs[i][j];
      datavecs[i][comp].push_back(data[i][j]);
    }
  }

  std::cout << "Sigma: \n" << Sigma << std::endl << std::endl;

  for (int h = 0; h < numComponents; h++) {
    std::cout << "### Component #" << h << std::endl;
    std::cout << "##### Atom: beta=" << betas[h].transpose()
              << ", sd=" << stddevs[h]
              << " weights per group: " << weights.col(h).transpose()
              << std::endl;
    std::cout << std::endl;
  }
}
