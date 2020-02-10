// #include <omp.h>
#include <string>

#include "src/collector.hpp"
#include "src/hdp.hpp"
#include "src/utils.hpp"

#include "univariate_mixture_state.pb.h"

using namespace std;

 /*
  * Main script, runs the MCMC simulations
  * Usage:
  * ./run_from_file.out \
  *     [path_to_data_file] \
  *     [path_to_outfile] \
  *
  * The input data file is a csv file with two columns, the first one is
  * the index of the spatial location, the second one is the datum.
  * The Header is assumed to be the first line
  *
  * w_mat is a csv file storing the proximity matrix W
  *
  */
int main(int ac, char* av[]) {

  // omp_set_dynamic(0);     // Explicitly disable dynamic teams
  // omp_set_num_threads(omp_get_num_threads() - 1);

  std::string infile = av[1];
  std::string outfile = av[2];
  std::cout << "infile: " << infile << std::endl;
  std::cout << "outfile: " << outfile << std::endl;


  std::vector<std::vector<double>> data = utils::readDataFromCSV(infile);

  std::cout << "NumGroups: " << data.size() << std::endl;

  int burnin = 10000;
  int niter = 10000;
  int thin = 10;

  std::deque<HdpState> chains;


  HdpSampler sampler(data);
  sampler.init();

  for (int i=0; i < burnin; i++) {
      sampler.sample();
  }

  for (int i=0; i < niter; i++) {
      sampler.sample();
      if ((i +1) % thin == 0)
          chains.push_back(sampler.getStateAsProto());
  }

  writeManyToFile(chains, outfile);

}
