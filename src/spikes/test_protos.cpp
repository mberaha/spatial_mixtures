#include <iostream>
#include <vector>
#include <string>
#include <google/protobuf/text_format.h>
// #include <google/protobuf/util/json_util.h>
#include "../protos/cpp/univariate_mixture_state.pb.h"
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <fstream>
#include <Eigen/Dense>
#include "../protos/cpp/eigen.pb.h"
#include <cstring>

using namespace google::protobuf;
using namespace google::protobuf::io;


void saveToFile(const std::vector<UnivariateMixtureState>& states,
                const std::string& filename) {
    std::ofstream outfile(filename, std::ofstream::out);
    if (outfile.is_open()) {
        for (UnivariateMixtureState state: states) {
            outfile << state.SerializeAsString() << std::endl;
        }
    } else {
        std::cout << "could not open file" << std::endl;
    }
    std::cout << "Done" << std::endl;
    outfile.close();
}

std::vector<UnivariateMixtureState> readFromFIle(const std::string& filename) {
    std::cout << "reading" << std::endl;
    std::vector<UnivariateMixtureState> out;
    std::ifstream infile(filename, std::ofstream::in);
    std::string line;
    if (infile.is_open()) {
        while (std::getline(infile, line)) {
            UnivariateMixtureState state;
            state.ParseFromString(line);
            // std::cout << state.DebugString() << std::endl;
            out.push_back(state);
        }
    } else {
        std::cout << "could not open file" << std::endl;
    }
    std::cout << "Done" << std::endl;
    infile.close();
    return out;
}

int main() {
    std::string s;
    UnivariateMixtureAtom* atom;
    // Now test serialization
    std::string outfile = "serialized_protos.dat";
    std::vector<UnivariateMixtureState> states;
    for (int i = 0; i < 10; i++) {
        UnivariateMixtureState curr;
        curr.set_num_components(i + 1);
        for (int j = 0; j < i+1; j++) {
            atom = curr.add_atoms();
            atom->set_mean(-1.0 * i);
            atom->set_stdev(0.5 * i);
        }
        states.push_back(curr);
    }


    saveToFile(states, "/home/mario/PhD/spatial_lda/src/chains.dat");

    std::vector<UnivariateMixtureState> restored = readFromFIle(
        "/home/mario/PhD/spatial_lda/src/chains.dat");

    std::cout << "Restored" << std::endl;
    for (int i=0; i < restored.size(); i++) {
        // std::cout << "##### State number " << i << std::endl;
        // restored[i].PrintDebugString();
    }

    Eigen::MatrixXd M(3,3);
    M << 1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0 ;
    std::cout<<M<<std::endl;

    EigenMatrix protomat;
    protomat.set_rows(3);
    protomat.set_cols(3);
    protomat.PrintDebugString();

    *protomat.mutable_data() = {M.data(), M.data() + M.size()};

    protomat.PrintDebugString();

    std::cout << "Bella lì" << std::endl;
    return 1;
}
