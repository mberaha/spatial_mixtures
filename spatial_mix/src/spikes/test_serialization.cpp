#include "../recordio.hpp"
#include "univariate_mixture_state.pb.h"
#include <deque>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    std::deque<UnivariateMixtureState> states;
    UnivariateMixtureAtom* atom;
    for (int i = 0; i < 5; i++) {
        UnivariateMixtureState curr;
        curr.set_num_components(i + 3);
        for (int j = 0; j < i+3; j++) {
            atom = curr.add_atoms();
            atom->set_mean(-1.0 * i);
            atom->set_stdev(0.5 * i);
        }
        states.push_back(curr);
    }

    writeManyToFile(
        states,
        "/home/mario/PhD/spatial_lda/spatial_mix/src/serialized.dat");

    std::deque<UnivariateMixtureState> restored;

    restored = readManyFromFile<UnivariateMixtureState>(
        "/home/mario/PhD/spatial_lda/spatial_mix/src/serialized.dat");

    std::cout << "Restred " << restored.size() << " messages" << std::endl;
    restored[1].PrintDebugString();

    std::cout << "Done" << std::endl;
    return 1;
}
