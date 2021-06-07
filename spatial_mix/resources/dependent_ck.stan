data {
    int<lower=1> num_groups;
    int<lower=1> ndata;
    int<lower=1> p;
    matrix[ndata, p] covariates;
    int<lower=1, upper=num_groups> obs2group[ndata];
    vector[ndata] y;
    int num_components;
    matrix[num_groups, num_groups] G;
    // parameters
    real a;
    real b;
    // spatial strength
    real rho;
}

transformed data {
    vector[num_components] mh;
    matrix[num_groups, num_groups] Lambda;
    matrix[num_groups, num_groups] Lambda_chol;
    vector[num_groups] sample_sigma;
    vector[num_groups] sample_mean;
    vector[num_groups] rowsums;

    // prior for psi
    real psi0;
    real nu0;
    // prior for \mu
    real mu0;
    real alpha;

    for (h in 1:num_components) {
        mh[h] = log(1.0 - 1.0 / (1 + exp(b - a * h)));
    }

    for (i in 1:num_groups)
        rowsums[i] = sum(G[i]);

    Lambda = diag_matrix(rowsums) - G * rho;
    for (i in 1:num_groups)
        print(Lambda[i]);

    print(determinant(Lambda));
}

parameters {
    vector[num_components] means;
    vector<lower=0.01>[num_components] vars;
    real<lower=0.01> psi;
    vector[num_groups] trans_weights[num_components];
    vector[p] beta;
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

    vector[ndata] xbeta = covariates * beta;


    // priors
    vars ~ inv_gamma(2, 2);
    for (h in 1:num_components) {
        means[h] ~ normal(0.0, sqrt(2 * vars[h]));
    }

    for (h in 1:num_components) {
        trans_weights[h] ~ multi_normal_prec(
            rep_vector(mh[h], num_groups), 0.5 * Lambda);
    }

    beta ~ normal(0, 2.5);


    // likelihood
    for (i in 1:ndata) {
        vector[num_components] contributions;

        for (h in 1:num_components) {
            contributions[h] = log(weights[obs2group[i]][h]) + normal_lpdf(
                    y[i] | xbeta[i] + means[h], sqrt(vars[h]));
        }
        target += log_sum_exp(contributions);
    }
}