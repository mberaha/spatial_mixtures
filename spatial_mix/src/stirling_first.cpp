#include "stirling_first.hpp"

unsigned long int stirling_first(int n, int m) {
  static stirlingmemoizer memo(stirling_);
  return memo(stirling_, n, m);
}
