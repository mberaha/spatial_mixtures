#ifndef SRC_COLLECTOR_HPP
#define SRC_COLLECTOR_HPP

#include <vector>
#include <list>
#include <fstream>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <cstdio>
#include <cstring>
#include <cerrno>

using namespace google::protobuf::io;

template<typename T>
class Collector {
 protected:
     bool in_memory;
     std::string filename;
     std::list<T> chains;
     std::ofstream fout;

 public:
     Collector(std::string filename): filename(filename) {
         in_memory = false;
     }

     Collector(int max_num_iter) {
         in_memory = true;
         chains.reserve(max_num_iter);
     }

     ~Collector() {}

     void collect(T state) {
         if (in_memory) {
             chains.push_back(state);
         } else {
             fout.open(filename, std::ios::out | std::ios::app);
             if (fout.fail())
                throw std::ios_base::failure(std::strerror(errno));

            fout.exceptions(
                fout.exceptions() | std::ios::failbit | std::ifstream::badbit);

            fout << state.SerializeToString() << std::endl;
            fout.close();
         }
     }

     void saveToFile(std::string fname) {
         // Writes the whole chunk
         fout.open(fname, std::ios::out);
         if (fout.fail())
            throw std::ios_base::failure(std::strerror(errno));

        fout.exceptions(
            fout.exceptions() | std::ios::failbit | std::ifstream::badbit);

         for (T state: chains) {
             fout << state.SerializeToString() << std::endl;
         }

        fout.close();
     }

     void loadFromFile(std::string filename) {
         std::ifstream infile(filename, std::ofstream::in);
         if (infile.fail())
            throw std::ios_base::failure(std::strerror(errno));

        std::string line;
        while (std::getline(infile, line)) {
            T state;
            state.ParseFromString(line);
            chains.push_back(state);
        }
     }
};

#endif
