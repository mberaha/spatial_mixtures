#include <vector>
#include <iostream>
#include <fstream>
#include <stan/math/prim/prob.hpp>
#include "../sampler.hpp"
#include "../collector.hpp"
#include "../recordio.hpp"
#include <random>
#include "univariate_mixture_state.pb.h"
#include "sampler_params.pb.h"


int main() {
    std::cout << "Beginning" << std::endl;
    std::mt19937_64 rng;


    int numGroups = 5;
    Eigen::VectorXd samples = Eigen::VectorXd::Zero(numGroups);
    int numSamples = 1000;
    std::cout << "numSamples: " << numSamples << std::endl;
    std::vector<std::vector<double>> data(numGroups);
    std::vector<Eigen::MatrixXd> regressors(numGroups);

    std::vector <Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>> metrics(2);
    std::vector<std::vector<int>> _neigh(numGroups);

    samples[0] = numSamples;
    samples[1] = numSamples;
    samples[2] = numSamples;
    samples[3] = numSamples;
    samples[4] = numSamples;

    _neigh[0]={1,2,3,4};
    _neigh[1]={0,2,3,4};
    _neigh[2]={0,1,3,4};
    _neigh[3]={0,1,2,4};
    _neigh[4]={0,1,2,3};

    metrics[0]= Eigen::MatrixXd::Zero(numGroups,numGroups).array();
    metrics[1]= Eigen::MatrixXd::Zero(numGroups,numGroups).array();

    Eigen::VectorXd beta(4);
    beta << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd sigma = Eigen::MatrixXd::Identity(4, 4);
    Eigen::VectorXd x_mean = Eigen::VectorXd::Zero(4);

    for (int i=0; i < numGroups ; i++) {
        data[i].resize(samples[i]);
        regressors[i].resize(samples[i], 4);
        if (i<2){
            std::cout << "Creo il gruppo: " << i <<
            " dalla prima distrizione "<<std::endl;

            for (int j=0; j < samples[i]; j++) {
                Eigen::VectorXd x = stan::math::multi_normal_rng(x_mean, sigma, rng);
                double err;
                double nu = 6;
                double mu = -2;
                double t_sigma = 1;

                err = stan::math::student_t_rng(nu, mu, t_sigma, rng);

                data[i][j] = x.transpose() * beta + err;
                regressors[i].row(j) = x;
            }
        }
        else {
            if (i==4){
              std::cout << "Creo il gruppo: " << i <<
              " dalla seconda distrizione "<<std::endl;
              for (int j=0; j < samples[i]; j++) {
                  Eigen::VectorXd x = stan::math::multi_normal_rng(x_mean, sigma, rng);
                  double err;
                  double ch_mu = 0;
                  double ch_sigma = 1;
                  err = stan::math::cauchy_rng(ch_mu, ch_sigma, rng);

                  data[i][j] = x.transpose() * beta + err;
                  regressors[i].row(j) = x;
              }
            }

            else {
              std::cout << "Creo il gruppo: " << i <<
              " dalla terza distribuzione "<<std::endl;
              double skew_mean = 2;
              double omega = 1;
              double alpha = 4;

              for (int j=0; j < samples[i]; j++) {
                  Eigen::VectorXd x = stan::math::multi_normal_rng(x_mean, sigma, rng);
                  double err;
                  err = stan::math::skew_normal_rng(skew_mean, omega, alpha, rng);
                  data[i][j] = x.transpose() * beta + err;
                  regressors[i].row(j) = x;
              }
            }
        }
    }

    // data[numGroups - 1].resize(numSamples);
    // regressors[numGroups - 1].resize(numSamples, 4);
    // for (int j=0; j < numSamples; j++) {
    //     Eigen::VectorXd x = stan::math::multi_normal_rng(x_mean, sigma, rng);
    //     double err;
    //     err = stan::math::normal_rng( 11.5, 1.0, rng);
    //     data[numGroups - 1][j] = x.transpose() * beta + err;
    //     regressors[numGroups - 1].row(j) = x;
    // }

    std::ofstream outfile;
    outfile.open("data_simulation_1.csv");
    outfile << "Group,Data" << std::endl;
    for (int i = 0; i < numGroups; i++) {
        for (int j=0; j < data[i].size(); j++) {
            outfile << i << "," << data[i][j] << std::endl;
        }
    }


    std::cout << "Initialized Data" << std::endl;
    // for (int i=0; i < numGroups; i++) {
    //     for (double d: data[i])
    //         std::cout << d << ", ";
    //     std::cout << "\n\n";
    // }

    std::deque<UnivariateState> chains;
    Eigen::MatrixXd W(numGroups, numGroups);
    W << 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0;
    // Eigen::MatrixXd W = Eigen::MatrixXd::Zero(3, 3);

    SamplerParams params = loadTextProto<SamplerParams>(
        "/home/mario/PhD/spatial_lda/spatial_mix/resources/sampler_params.asciipb");
    std::cout << params.DebugString() << std::endl;

    SpatialMixtureSampler spSampler(params, data, W);
    std::cout<<"Init start"<<std::endl;
    spSampler.init();
    std::cout<<"Init done"<<std::endl;
    spSampler.printDebugString();

    std::cout<<"Starting sampling"<<std::endl;
    for (int i=0; i < 1000; i++) {
        spSampler.sample();
    }
    spSampler.printDebugString();
    for (int i=0; i < 1000; i++) {
        spSampler.sample();
        if ((i+1) % 10 == 0) {
            chains.push_back(spSampler.getStateAsProto());
        }
    }

    spSampler.printDebugString();
    // writeManyToFile(chains, "chains_now2.dat");
    // std::cout << "Done" << std::endl;
    //
    //
    // std::deque<UnivariateState> restored;
    // restored = readManyFromFile<UnivariateState>("chains_now2.dat");
    //
    // std::cout << "******** RESTORED ********" << std::endl;
    //
    // std::cout << "Acceptance rate for Rho: " <<
    //     1.0 * spSampler.getNumAccepted() / (1.0 * 15000) << std::endl;

    return 1;
}
