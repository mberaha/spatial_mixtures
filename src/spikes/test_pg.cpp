#include <iostream>
#include "PolyaGammaHybrid.h"

int main() {
    unsigned long int seed = 1231232141241;
    PolyaGammaHybridDouble pg_rng(seed);

    for (int i=0; i < 10; i++) {
        std::cout << pg_rng.draw(1.0, 3.0) << std::endl;
    }
}
