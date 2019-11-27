#ifndef SRC_UTILS_HPP
#define SRC_UTILS_HPP

#include <Eigen/Dense>

namespace utils {

Eigen::VectorXd Alr(Eigen::VectorXd x);

Eigen::VectorXd InvAlr(Eigen::VectorXd x);

} // namespace utils

#endif // SRC_UTILS_HPP
