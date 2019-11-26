#include <vector>
#include <stan/math.hpp>
#include <numeric>
#include <algorithm>
#include <random>
#include <iostream>

std::vector<double> normalGammaUpdate(
    std::vector<double> data, double priorMean, double priorA,
    double priorB, double priorLambda) {
  double postMean, postA, postB, postLambda;
  int n = data.size();

  double sum = std::accumulate(std::begin(data), std::end(data), 0.0);
  double ybar = sum / n;
  postMean = (priorLambda * priorMean + sum) / (priorLambda + n);
  postA = 1.0 * priorA + 1.0 * n / 2;

  double ss = 0.0;
  std::for_each(data.begin(), data.end(), [&ss, &ybar](double x) {
    ss += (x - ybar) * (x - ybar);});

  postB = (
      priorB + 0.5 * ss +
      0.5 * priorLambda / (n + priorLambda) * n *(ybar - priorMean) * (ybar - priorMean));

  postLambda = priorLambda + n;

  return std::vector<double>{postMean, postA, postB, postLambda};
}

int main() {
    std::mt19937_64 rng{std::random_device()()};
    // first test the gamma RNG
    for (int i=0; i < 5; i++)
        std::cout << stan::math::gamma_rng(100, 5, rng) << std::endl;

    // simulate some data
    std::vector<double> xx(100);
    for (int i=0; i < 100; i++)
        xx[i] = stan::math::normal_rng(10, 1, rng);

    // look at the posterior parameters
    std::vector<double> temp = normalGammaUpdate(xx, 10.0, 2.0, 2.0, 10.0);
    std::cout << "Posterior Mean: " << temp[0] << std::endl;
    std::cout << "Posterior A: " << temp[1] << std::endl;
    std::cout << "Posterior B: " << temp[2] << std::endl;
    std::cout << "Posterior Lambda: " << temp[3] << std::endl;


    // generate a draw from the posterior
    double tau = stan::math::gamma_rng(temp[1], temp[2], rng);
    double mu = stan::math::normal_rng(temp[0], 1.0 / std::sqrt(temp[3] * tau), rng);

    std::cout << "Posterior Mean: " << mu << std::endl;
    std::cout << "Posterior SD: " << 1.0 / tau << std::endl;

    return 1;
}
