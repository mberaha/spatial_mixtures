#ifndef SRC_UTILS_HPP
#define SRC_UTILS_HPP

#include <map>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <Eigen/Dense>

namespace utils {

Eigen::VectorXd Alr(Eigen::VectorXd x);

Eigen::VectorXd InvAlr(Eigen::VectorXd x);

std::vector<std::vector<double>> readDataFromCSV(std::string filename);

Eigen::MatrixXd readMatrixFromCSV(std::string filename);

} // namespace utils

#endif // SRC_UTILS_HPP
