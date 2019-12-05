#include <string>
#include <boost/program_options.hpp>

#include "src/collector.hpp"
#include "src/sampler.hpp"
#include "src/utils.hpp"

#include "univariate_mixture_state.pb.h"

namespace po = boost::program_options;
using namespace std;


po::variables_map getArgs(int ac, char*av[]) {
   po::options_description desc("Allowed options");
     po::variables_map vm;
   try {
     desc.add_options()
         ("help", "produce help message")
         ("input", po::value<string>(), "")
         ("ngroups", po::value<int>())
         ("w_mat", po::value<int>())
         ("outfile", po::value<string>(), "");

     po::store(po::parse_command_line(ac, av, desc), vm);
     po::notify(vm);
   }
   catch(exception& e) {
       cerr << "error: " << e.what() << "\n";
   }
   catch(...) {
       cerr << "Exception of unknown type!\n";
   }

   return vm;
 }

 /*
  * Main script, runs the MCMC simulations
  * Usage:
  * ./run_from_file.out \
  *     --input [path_to_data_file] \
  *     --w_mat [path_to_w_mat_file] \
  *     --outfile [path_to_outfile] \
  *
  * The input data file is a csv file with two columns, the first one is
  * the index of the spatial location, the second one is the datum.
  * The Header is assumed to be the first line
  *
  * w_mat is a csv file storing the proximity matrix W
  *
  */
int main(int ac, char* av[]) {
  po::variables_map args = getArgs(ac, av);
  std::string outfile = args["outfile"].as<string>();
  Eigen::MatrixXd W = utils::readMatrixFromCSV(args["w_mat"].as<string>());
  std::vector<std::vector<double>> data = utils::readDataFromCSV(
    args["input"].as<string>());

  int burnin = 10000;
  int niter = 10000;
  int thin = 10;

  Collector<UnivariateState> collector(niter / thin + 10);

  SpatialMixtureSampler spSampler(data, W);
  spSampler.init();

  for (int i=0; i < burnin; i++) {
      spSampler.sample();
  }

  for (int i=0; i < niter; i++) {
      spSampler.sample();
      if (i % thin == 0)
          spSampler.saveState(&collector);
  }

  collector.saveToFile(outfile);

}
