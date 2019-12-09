#include "../utils.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {

    Eigen::VectorXd x(4);
    x << 1, 2, 3, 4;
    x /= x.sum();


    std::cout << "a" << std::endl;

    std::cout << "x: " << x.transpose() << std::endl;
    std::cout << "alr(x): " << utils::Alr(x).transpose() << std::endl;
    std::cout << "x: " << utils::InvAlr(utils::Alr(x)).transpose() << std::endl;

    Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 3);
    Eigen::MatrixXd E = 0.5*(A+A.transpose()+Eigen::MatrixXd::Identity(3, 3));
    Eigen::MatrixXd inv = E.llt().solve(Eigen::MatrixXd::Identity(3, 3));
    std::cout<<E*inv<<std::endl<<std::endl;


    std::cout<<A<<std::endl<<std::endl;
    std::cout<<utils::removeRowColumn(A, 1)<<std::endl<<std::endl;
    std::cout<<A<<std::endl<<std::endl;
    double out = A.row(0)*A.col(0);
    std::cout<<A.row(0)*A.col(0)<<std::endl<<std::endl;
    return 1;
}
