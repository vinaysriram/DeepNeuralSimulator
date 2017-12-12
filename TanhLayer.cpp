
#include <cmath>
#include <iostream>
#include "TanhLayer.hpp"

using namespace Eigen;
using namespace std;

TanhLayer::TanhLayer(int numNeurons, int numInputs, default_random_engine &gen)
  : AbstractLayer(numNeurons, numInputs, gen) {}

TanhLayer::~TanhLayer() {}

void TanhLayer::activation(VectorXd &input, VectorXd &output)
{
  for(int i = 0; i < input.size(); i++) {
    output(i) = tanh(input(i));
  }
}

void TanhLayer::gradient(VectorXd &input, VectorXd &output)
{
  for(int i = 0; i < input.size(); i++) {
    output(i) = 1 - pow(tanh(input(i)), 2);
  }
}
