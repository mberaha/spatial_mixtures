#include <iostream>
#include <vector>
#include <string>
// #include <google/protobuf/text_format.h>
// #include <google/protobuf/util/json_util.h>
#include "../protos/cpp/univariate_mixture_state.pb.h"

using namespace google::protobuf;

int main() {
    std::cout << "ciao" << std::endl;
    std::string s;

    UnivariateMixtureState state;
    state.set_num_components(1);

    UnivariateMixtureAtom* atom;
    atom = state.add_atoms();
    atom->set_mean(3.0);
    atom->set_stdev(10);

    atom = state.add_atoms();
    atom->set_mean(5.0);
    atom->set_stdev(1.0);

    atom = state.add_atoms();
    atom->set_mean(-3.0);
    atom->set_stdev(0.5);

    std::cout << "a" << std::endl;
    state.PrintDebugString();
    std::cout << "b" << std::endl;
    std::cout << "Number of atoms: " << state.atoms_size() << std::endl;
    for (int i=0; i < state.atoms_size(); i++) {
        std::cout << "Atom #" << i << "; mean:" << state.atoms()[i].mean() << std::endl;
    }

    // Now test serialization
    std::string outfile = "serialized_protos.rio";
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

    // for (int i = 0; i < 10; i++) {
    //     std::cout << "##### State number " << i << std::endl;
    //     states[i].PrintDebugString();
    // }

    return 1;
}
