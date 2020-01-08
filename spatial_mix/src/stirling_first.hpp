//
// Created by mario on 31/07/19.
//

#ifndef CPPMODEL_STIRLING_NUMBERS_HPP
#define CPPMODEL_STIRLING_NUMBERS_HPP

#include <vector>

/*
 * Memoizer specific to the stirling numbers: instead of using
 * a map, employs a lower triangular matrix to gain efficiency.
 */
class stirlingmemoizer {
 private:
   std::vector<std::vector<unsigned long int>> memoMatrix;
   int maxN;
   int maxM;

 public:
   template <class F>
   stirlingmemoizer(F f) {
     maxN = maxM = 100;
     memoMatrix.resize(maxN + 1);
     for (int n = 0; n <= maxN; n++) {
       memoMatrix[n].resize(n + 1);
       memoMatrix[n][0] = 0;
       for (int m = 1; m <= n; m++) {
         memoMatrix[n][m] = (n - 1) * memoMatrix[n - 1][m] + memoMatrix[n - 1][m - 1];
       }
     }
   }

   template <class F>
   void addRows(F f, int newMaxN) {
     memoMatrix.resize(newMaxN + 1);
     for (int n = maxN; n <= newMaxN; n++) {
       memoMatrix[n].resize(n + 1);
       memoMatrix[n][0] = 0;
       for (int m = 1; m <= n; m++) {
         memoMatrix[n][m] = (n - 1) * memoMatrix[n - 1][m] + memoMatrix[n - 1][m - 1];
       }
     }
     maxN = newMaxN;
   }

   template <class F>
   const unsigned long int& operator()(F f, int n, int m) {
     if (n > maxN)
       addRows(f, n);
     return memoMatrix[n][m];
   }

};

unsigned long int StirlingFirst(int n, int m);



namespace {
unsigned long int stirling_(int n, int m) {
    if (((n == 0) & (m == 0)) ||( (n == 1) & (m == 1)))
      return 1;
    else if ((n > 0) & (m == 0))
      return 0;
    else if (m > n)
      return 0;
    else
      return StirlingFirst(n-1, m-1) + (n-1) * StirlingFirst(n-1, m);
}
}

unsigned long int StirlingFirst(int n, int m) {
  static stirlingmemoizer memo(stirling_);
  return memo(stirling_, n, m);
}


#endif  // CPPMODEL_STIRLING_NUMBERS_HPP
