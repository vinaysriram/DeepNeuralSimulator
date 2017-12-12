
#include <cmath>
#include <iostream>
#include "ReLuLayer.hpp"

using namespace Eigen;
using namespace std;

ReLuLayer::ReLuLayer(int numNeurons, int numInputs, default_random_engine &gen)
  : AbstractLayer(numNeurons, numInputs, gen) {}

ReLuLayer::~ReLuLayer() {}

void ReLuLayer::activation(VectorXd &input, VectorXd &output)
{
  for(int i = 0; i < input.size(); i++) {
    output(i) = (input(i) > 0) ? input(i) : 0;
  }
}

void ReLuLayer::gradient(VectorXd &input, VectorXd &output)
{
  for(int i = 0; i < input.size(); i++) {
    output(i) = (input(i) > 0) ? 1 : 0;
  }
}
