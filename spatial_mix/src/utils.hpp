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

Eigen::MatrixXd removeRow(Eigen::MatrixXd matrix, unsigned int rowToRemove);

Eigen::MatrixXd removeColumn(Eigen::MatrixXd matrix, unsigned int colToRemove);

Eigen::MatrixXd removeRowColumn(Eigen::MatrixXd matrix, unsigned int toRemove);

} // namespace utils

#endif // SRC_UTILS_HPP
