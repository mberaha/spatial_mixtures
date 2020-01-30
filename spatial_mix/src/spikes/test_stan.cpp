#include <iostream>
#include <stan/math/prim/mat.hpp>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

int main() {
    // Eigen::VectorXd mean(2);
    // mean << 0, 10;

    std::mt19937_64 rng;

    // Eigen::MatrixXd var(2, 2);
    // var << 1, 0.5, 0.5, 1;
    //
    // Eigen::VectorXd normalMat = stan::math::multi_normal_rng(mean, var, rng);
    // std::cout << "normalMat = \n"
    //       << normalMat
    //       << std::endl;
    //
    // double x =  stan::math::normal_rng(0.0, 1.0, rng);
    // std::cout << "x = " << x << std::endl;

    Eigen::MatrixXd sigma = Eigen::MatrixXd::Identity(4, 4);

    Eigen::VectorXd meanvec = Eigen::VectorXd::Zero(5 * 4);
    Eigen::MatrixXd w(5, 4);
    double rho = 0.9;

    w << 1.97219, -3.07006, -1.87972,  2.25326,
         2.01784, -2.01608, -2.73986,  2.16515,
         3.19236, -1.29181, 0.041094,  3.12665,
         -3.32552,  1.65446,  2.97848, -1.41129,
         -2.31945,   1.9644,  3.30069, -1.29228;

    std::cout << "w: \n" << w << std::endl;
    w = w.transpose().eval();
    Eigen::VectorXd vectorizedW = Eigen::Map<Eigen::VectorXd>(w.data(), w.size());
    std::cout << "vectorizedW: \n" << vectorizedW << std::endl;

    Eigen::MatrixXd prec1_ = Eigen::MatrixXd::Zero(5, 5);
    for (int i=0; i < 3; i++) {
        prec1_(i, i) = 2.0;
        for (int j=0; j < i; j++) {
            prec1_(i, j) = -rho;
            prec1_(j, i) = -rho;
        }
    }
    for (int i = 3; i < 5; i++) {
        prec1_(i, i) = 1.0;
        for (int j=3; j < i; j++) {
            prec1_(i, j) = -rho;
            prec1_(j, i) = -rho;
        }
    }
    std::cout << "prec1_: \n" << prec1_ << std::endl;
    Eigen::MatrixXd prec1 = kroneckerProduct(prec1_, sigma);


    Eigen::MatrixXd prec2_ = Eigen::MatrixXd::Zero(5, 5);
    for (int i=0; i < 5; i++) {
        prec2_(i, i) = 4.0;
        for (int j=0; j < i; j++) {
            prec2_(i, j) = -rho;
            prec2_(j, i) = -rho;
        }
    }

    std::cout << "prec2_: \n" << prec2_ << std::endl;
    Eigen::MatrixXd prec2 = kroneckerProduct(prec2_, sigma);


    Eigen::MatrixXd prec3_ = Eigen::MatrixXd::Zero(5, 5);
    prec3_ <<  1, 0, 0, 0, -0.99,
               0, 1, 0, 0, 0,
               0, 0, 1, 0, -0.99,
               0, 0, 0, 1, -0.99,
              -0.99, 0, -0.99, -0.99, 3;

    Eigen::MatrixXd prec3 = kroneckerProduct(prec3_, sigma);
    std::cout << "prec3_: \n" << prec3_ << std::endl;

    Eigen::MatrixXd prec4_ = prec3_;
    prec4_(3, 4) = 0;
    prec4_(4, 3) = 0;
    prec4_(4, 4) = 2;
    std::cout << "prec4_: \n" << prec4_ << std::endl;

    Eigen::MatrixXd prec4 = kroneckerProduct(prec4_, sigma);

    // Eigen::MatrixXd prec3 = prec2_;
    // prec3(0, 3) = 0;
    // prec3(3, 0) = 0;
    // prec3(0, 0) -= 1;
    // prec3(3, 3) -= 1;
    // std::cout << "prec3: \n" << prec3 << std::endl;

    std::cout << "With prec1: " << stan::math::multi_normal_prec_lpdf(
        vectorizedW, meanvec, prec1) << std::endl;

    std::cout << "With prec2: " << stan::math::multi_normal_prec_lpdf(
        vectorizedW, meanvec, prec2) << std::endl;

    std::cout << "With prec3: " << stan::math::multi_normal_prec_lpdf(
        vectorizedW, meanvec, prec3) << std::endl;

    std::cout << "With prec4: " << stan::math::multi_normal_prec_lpdf(
        vectorizedW, meanvec, prec4) << std::endl;


    // std::cout << "With prec3: " << stan::math::matrix_normal_prec_lpdf(
    //     w, mean, prec3, sigma) << std::endl;
    return 1;
}
