#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <deque>
#include <string>
#include <vector>

#include "src/dependent_sampler.hpp"
#include "src/hdp.hpp"
#include "src/sampler.hpp"
#include "src/utils.hpp"
#include "univariate_mixture_state.pb.h"

namespace py = pybind11;

using return_t = std::tuple<std::deque<py::bytes>, double>;

return_t _runSpatialSampler(int burnin, int niter, int thin,
                            const std::vector<std::vector<double>> &data,
                            const Eigen::MatrixXd &W,
                            const SamplerParams &params,
                            const std::vector<Eigen::MatrixXd> &covariates) {
  SpatialMixtureSampler spSampler(params, data, W, covariates);
  spSampler.init();

  std::deque<py::bytes> out;
  int log_every = 200;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < burnin; i++) {
    spSampler.sample();
    if ((i + 1) % log_every == 0)
      py::print("Burn-in, iter #", i + 1, " / ", burnin);
  }

  for (int i = 0; i < niter; i++) {
    spSampler.sample();
    if ((i + 1) % thin == 0) {
      std::string s;
      spSampler.getStateAsProto().SerializeToString(&s);
      out.push_back((py::bytes)s);
    }
    if ((i + 1) % log_every == 0)
      py::print("Running, iter #", i + 1, " / ", niter);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - start).count();

  return std::make_tuple(out, duration);
}

return_t runSpatialSamplerPythonFromFiles(
    int burnin, int niter, int thin, std::string infile, std::string w_file,
    std::string params_file, const std::vector<Eigen::MatrixXd> &covariates) {
  Eigen::MatrixXd W = utils::readMatrixFromCSV(w_file);
  std::vector<std::vector<double>> data = utils::readDataFromCSV(infile);
  SamplerParams params = loadTextProto<SamplerParams>(params_file);
  return _runSpatialSampler(burnin, niter, thin, data, W, params, covariates);
}

return_t runSpatialSamplerPythonFromData(
    int burnin, int niter, int thin,
    const std::vector<std::vector<double>> &data, const Eigen::MatrixXd &W,
    std::string serialized_params,
    const std::vector<Eigen::MatrixXd> &covariates) {
  SamplerParams params;
  params.ParseFromString(serialized_params);
  return _runSpatialSampler(burnin, niter, thin, data, W, params, covariates);
}

return_t runHdpPythonFromData(int burnin, int niter, int thin,
                              const std::vector<std::vector<double>> &data) {
  HdpSampler sampler(data);
  sampler.init();

  std::deque<py::bytes> out;
  int log_every = 200;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < burnin; i++) {
    sampler.sample();
    if ((i + 1) % log_every == 0)
      py::print("Burn-in, iter #", i + 1, " / ", burnin);
  }

  for (int i = 0; i < niter; i++) {
    sampler.sample();
    if ((i + 1) % thin == 0) {
      std::string s;
      sampler.getStateAsProto().SerializeToString(&s);
      out.push_back((py::bytes)s);
    }
    if ((i + 1) % log_every == 0)
      py::print("Running, iter #", i + 1, " / ", niter);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - start).count();

  return std::make_tuple(out, duration);
}

return_t runDependentPython(int burnin, int niter, int thin,
                            const std::vector<std::vector<double>> &data,
                            const Eigen::MatrixXd &W,
                            std::string serialized_params,
                            const std::vector<Eigen::MatrixXd> &covariates) {
  SamplerParams params;
  params.ParseFromString(serialized_params);

  DependentSpatialMixtureSampler spSampler(params, data, W, covariates);
  spSampler.init();

  std::deque<py::bytes> out;
  int log_every = 200;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < burnin; i++) {
    spSampler.sample();
    if ((i + 1) % log_every == 0)
      py::print("Burn-in, iter #", i + 1, " / ", burnin);
  }

  for (int i = 0; i < niter; i++) {
    spSampler.sample();
    if ((i + 1) % thin == 0) {
      std::string s;
      spSampler.getStateAsProto().SerializeToString(&s);
      out.push_back((py::bytes)s);
    }
    if ((i + 1) % log_every == 0)
      py::print("Running, iter #", i + 1, " / ", niter);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - start).count();

  return std::make_tuple(out, duration);
}

PYBIND11_MODULE(spmixtures, m) {
  m.doc() = "aaa";  // optional module docstring

  m.def(
      "runSpatialSamplerFromFiles", &runSpatialSamplerPythonFromFiles,
      "runs the spatial sampler, returns a list (deque) of serialized protos");

  m.def(
      "runSpatialSamplerFromData", &runSpatialSamplerPythonFromData,
      "runs the spatial sampler, returns a list (deque) of serialized protos");

  m.def("runHdpPythonFromData", &runHdpPythonFromData,
        "runs the HDP sampler, returns a list (deque) of serialized protos");

  m.def("runDependentFromData", &runDependentPython,
        "runs the Dependent sampler, returns a list (deque) of serialized "
        "protos");
}
