
#include "Network.hpp"
#include <iostream>
#include <random>
#include <vector>

using namespace Eigen;
using namespace std;

int main(int argc, char **argv)
{
  /* Inputs and Outputs */
  Eigen::initParallel();

  VectorXi dims(3);
  VectorXi types(1);

  dims(0) = 2;
  dims(1) = 1000;
  dims(2) = 1;
  types(0) = 2;
  
  Network a(dims, types, argv[1], true, 0.01, 0);
  
  /* Compute Outputs */
  VectorXd input(2);
  VectorXd output(1);
  for(int i = -10; i <= 10; i++) {
    for(int j = -10; j < 10; j++) {
      input(0) = i;
      input(1) = j;
      a.runNetwork(input, output);
      cout << output(0) << ", ";
    }
    input(0) = i;
    input(1) = 10;
    a.runNetwork(input, output);
    cout << output(0) << endl;
    cerr << "Network run on " << i;
    cerr << " test rows." << endl;
  }  
}
