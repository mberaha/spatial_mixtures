#include "utils.hpp"

namespace utils {

Eigen::VectorXd Alr(Eigen::VectorXd x) {
    int D = x.size();
    Eigen::VectorXd out = x.head(D-1);
    out /= x(D-1);
    out = out.array().log();
    return out;
}

Eigen::VectorXd InvAlr(Eigen::VectorXd x) {
    int D = x.size() + 1;
    Eigen::VectorXd out(D);
    out.head(D - 1) = x;
    out(D - 1) = 0;
    out = out.array().exp();
    return out / out.sum();
}

}
