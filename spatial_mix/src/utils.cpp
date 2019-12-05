#include "utils.hpp"

namespace utils {

Eigen::VectorXd Alr(Eigen::VectorXd x) {
    int D = x.size();
    Eigen::VectorXd out = x.head(D-1);
    out /= x(D-1);
    out = out.array().log();
    return out;
}

Eigen::VectorXd InvAlr(Eigen::VectorXd x) {
    int D = x.size() + 1;
    Eigen::VectorXd out(D);
    out.head(D - 1) = x;
    out(D - 1) = 0;
    out = out.array().exp();
    return out / out.sum();
}

std::vector<std::vector<double>> readDataFromCSV(std::string filename) {
    std::ifstream infile(filename);

    std::map<int, std::vector<double>> out;

    int group;
    double datum;
    std::string line;
    char delim;

    // skip header
    std::getline(infile, line);

    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      if (!(iss >> group >> delim >> datum) ) { break; }
      out[group - 1].push_back(datum);
    }

    // maps are sorted
    int ngroups = out.rbegin()->first;
    bool startsFromZero = out.begin()->first == 0;

    std::vector<std::vector<double>> data(ngroups);
    for (int g=0; g < ngroups; g++) {
        if (startsFromZero)
            data[g] = out[g];
        else
            data[g] = out[g + 1];
    }

    return data;
}

Eigen::MatrixXd readMatrixFromCSV(std::string filename) {
    int MAXBUFSIZE = ((int) 1e6);
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(filename);
    while (! infile.eof())
        {
        std::string line;
        getline(infile, line);

        int temp_cols = 0;
        std::stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
        }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    Eigen::MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;

}

}
