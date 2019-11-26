#include "../utils.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {

    Eigen::VectorXd x(4);
    x << 1, 2, 3, 4;
    x /= x.sum();


    std::cout << "a" << std::endl;

    std::cout << "x: " << x.transpose() << std::endl;
    std::cout << "alr(x): " << alr(x).transpose() << std::endl;
    std::cout << "x: " << inv_alr(alr(x)).transpose() << std::endl;

    return 1;
}
