#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <deque>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "src/sampler.hpp"
#include "src/utils.hpp"

#include "univariate_mixture_state.pb.h"

namespace py = pybind11;

std::deque<py::bytes> runSpatialSamplerPython(
        int burnin, int niter, int thin,
        std::string infile, std::string w_file, std::string params_file) {

    Eigen::MatrixXd W = utils::readMatrixFromCSV(w_file);
    std::vector<std::vector<double>> data = utils::readDataFromCSV(infile);

    SamplerParams params = loadTextProto<SamplerParams>(params_file);
    SpatialMixtureSampler spSampler(params, data, W);
    spSampler.init();

    std::deque<py::bytes> out;

    for (int i=0; i < burnin; i++) {
        spSampler.sample();
    }

    for (int i=0; i < niter; i++) {
        spSampler.sample();
        if ((i +1) % thin == 0) {
            std::string s;
            spSampler.getStateAsProto().SerializeToString(&s);
            out.push_back((py::bytes) s);
        }
    }
    std::cout << "done" << std::endl;
    return out;
}



PYBIND11_MODULE(spmixtures, m) {
    m.doc() = "aaa"; // optional module docstring

    m.def("runSpatialSampler", &runSpatialSamplerPython,
          "runs the spatial sampler, returns a list (deque) of serialized protos");
}
