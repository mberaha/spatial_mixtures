data {
    int<lower=1> num_groups;
    int<lower=1> max_data_per_group;
    int num_data_per_group[num_groups];
    matrix[num_groups, max_data_per_group] data_by_group;
    int num_components;
    matrix[num_groups, num_groups] G;
    // parameters
    real a;
    real b;
    // spatial strength
    real rho;

    int points_in_grid;
    vector[points_in_grid] xgrid;
}

transformed data {
    vector[num_components] mh;
    matrix[num_groups, num_groups] Lambda;
    matrix[num_groups, num_groups] Lambda_chol;
    vector[num_groups] sample_sigma;
    vector[num_groups] sample_mean;

    // prior for psi
    real psi0;
    real nu0;
    // prior for \mu
    real mu0;
    real alpha;

    for (h in 1:num_components) {
        mh[h] = log(1.0 - 1.0 / (1 + exp(b - a * h)));
    }

    Lambda = diag_matrix(rep_vector(1.0, num_groups)) - G * rho;

    for (i in 1:num_groups) {
        sample_sigma[i] = sd(data_by_group[i][1:num_data_per_group[i]]);
        sample_mean[i] = mean(data_by_group[i][1:num_data_per_group[i]]);
    }
    psi0 = 0.7 * mean(sample_sigma) * pow(mean(to_vector(num_data_per_group)), -0.2);
    nu0 = 2;

    alpha = 0.7 * 0.7 * pow(num_groups, - 0.4);
    mu0 = mean(sample_mean);
}

parameters {
    vector[num_components] means;
    vector<lower=0.01>[num_components] vars;
    real<lower=0.01> psi;
    vector[num_groups] trans_weights[num_components];
}

transformed parameters {
    vector[num_components] stddevs;
    vector[num_components] weights[num_groups];

    for (h in 1:num_components)
        stddevs[h] = sqrt(vars[h]);

    for (i in 1:num_groups) {
        for (h in 1:num_components) {
            weights[i][h] = exp(trans_weights[h][i]);
        }
        weights[i] = weights[i] / sum(weights[i]);
    }
}


model {
    real tausq = 0.6 * 0.6;
    real nu = 2;

    vector[num_components] contributions[num_groups];

    // priors
    psi ~ gamma(nu0 / 2, nu0 / (2 * psi0));
    vars ~ inv_gamma(nu / 2, (nu * psi) / 2);
    for (h in 1:num_components) {
        means[h] ~ normal(mu0, sqrt(vars[h] / alpha));
    }

    for (h in 1:num_components) {
        trans_weights[h] ~ multi_normal_prec(
            rep_vector(mh[h], num_groups), tausq * Lambda);
    }

    // likelihood
    for (i in 1:num_groups) {
        for (j in 1:num_data_per_group[i]) {
            for (h in 1:num_components) {
                contributions[i][h] = log(weights[i][h]) + normal_lpdf(
                    data_by_group[i, j] | means[h], sqrt(vars[h]));
            }
        }
        target += log_sum_exp(contributions[i]);
    }
}