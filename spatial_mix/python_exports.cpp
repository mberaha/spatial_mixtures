#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <deque>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "src/sampler.hpp"
#include "src/utils.hpp"

#include "univariate_mixture_state.pb.h"

namespace py = pybind11;


std::deque<py::bytes> _runSpatialSampler(
        int burnin, int niter, int thin,
        const std::vector<std::vector<double>> &data,
        const Eigen::MatrixXd &W, const SamplerParams &params) {

    SpatialMixtureSampler spSampler(params, data, W);
    spSampler.init();

    std::deque<py::bytes> out;

    int log_every = 200;

    for (int i=0; i < burnin; i++) {
        spSampler.sample();
        if ((i + 1) % log_every == 0)
            std::cout << "Burn-in, iter #" << i+1 << " / " << burnin << std::endl;
    }

    for (int i=0; i < niter; i++) {
        spSampler.sample();
        if ((i +1) % thin == 0) {
            std::string s;
            spSampler.getStateAsProto().SerializeToString(&s);
            out.push_back((py::bytes) s);
        }
        if ((i + 1) % log_every == 0)
            std::cout << "Running, iter #" << i+1 << " / " << burnin << std::endl; 
    }
    return out;
}


std::deque<py::bytes> runSpatialSamplerPythonFromFiles(
        int burnin, int niter, int thin,
        std::string infile, std::string w_file, std::string params_file) {

    Eigen::MatrixXd W = utils::readMatrixFromCSV(w_file);
    std::vector<std::vector<double>> data = utils::readDataFromCSV(infile);
    SamplerParams params = loadTextProto<SamplerParams>(params_file);
    return _runSpatialSampler(burnin, niter, thin, data, W, params);
}


std::deque<py::bytes> runSpatialSamplerPythonFromData(
        int burnin, int niter, int thin,
        std::vector<std::vector<double>> data, Eigen::MatrixXd W,
        std::string serialized_params) {

    SamplerParams params;
    params.ParseFromString(serialized_params);
    return _runSpatialSampler(burnin, niter, thin, data, W, params);
}


PYBIND11_MODULE(spmixtures, m) {
    m.doc() = "aaa"; // optional module docstring

    m.def("_runSpatialSamplerFromFiles", &runSpatialSamplerPythonFromFiles,
          "runs the spatial sampler, returns a list (deque) of serialized protos");

    m.def("_runSpatialSamplerFromData", &runSpatialSamplerPythonFromData,
        "runs the spatial sampler, returns a list (deque) of serialized protos");

}
