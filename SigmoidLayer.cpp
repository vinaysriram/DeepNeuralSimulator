
#include <cmath>
#include <iostream>
#include "SigmoidLayer.hpp"

using namespace Eigen;
using namespace std;

SigmoidLayer::SigmoidLayer(int numNeurons, int numInputs, default_random_engine &gen)
  : AbstractLayer(numNeurons, numInputs, gen) {}

SigmoidLayer::~SigmoidLayer() {}

void SigmoidLayer::activation(VectorXd &input, VectorXd &output)
{
  for(int i = 0; i < input.size(); i++) {
    output(i) = 1.0 / (1.0 + exp(-input(i)));
  }
}

void SigmoidLayer::gradient(VectorXd &input, VectorXd &output)
{
  for(int i = 0; i < input.size(); i++) {
    output(i) = exp(input(i)) / pow(1.0 + exp(input(i)), 2);
  }
}
