
#include <cmath>
#include <iostream>
#include "SoftmaxLayer.hpp"

using namespace Eigen;
using namespace std;

SoftmaxLayer::SoftmaxLayer(int numNeurons, int numInputs, default_random_engine &gen)
  : AbstractLayer(numNeurons, numInputs, gen) {}

SoftmaxLayer::~SoftmaxLayer() {}

void SoftmaxLayer::activation(VectorXd &input, VectorXd &output)
{
  double sum = 0;
  for(int i = 0; i < input.size(); i++) {
    output(i) = exp(input(i));
    sum += output(i);
  }
  for(int i = 0; i < input.size(); i++) {
    output(i) = output(i) / sum;
  }
}

void SoftmaxLayer::gradient(VectorXd &input, VectorXd &output)
{
  output = VectorXd::Constant(input.size(), 1);
}
