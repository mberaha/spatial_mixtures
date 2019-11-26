#include <iostream>
#include <stan/math.hpp>
#include <random>

int main() {
    Eigen::VectorXd mean(2);
    mean << 0, 10;

    std::mt19937_64 rng;

    Eigen::MatrixXd var(2, 2);
    var << 1, 0.5, 0.5, 1;

    Eigen::VectorXd normalMat = stan::math::multi_normal_rng(mean, var, rng);
    std::cout << "normalMat = \n"
          << normalMat
          << std::endl;

    double x =  stan::math::normal_rng(0.0, 1.0, rng);
    std::cout << "x = " << x << std::endl;
    return 1;
}
