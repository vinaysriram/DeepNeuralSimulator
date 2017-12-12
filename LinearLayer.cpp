
#include <cmath>
#include <iostream>
#include "LinearLayer.hpp"

using namespace Eigen;
using namespace std;

LinearLayer::LinearLayer(int numNeurons, int numInputs, default_random_engine &gen)
  : AbstractLayer(numNeurons, numInputs, gen) {}

LinearLayer::~LinearLayer() {}

void LinearLayer::activation(VectorXd &input, VectorXd &output)
{
  output = input;
}

void LinearLayer::gradient(VectorXd &input, VectorXd &output)
{
  output = VectorXd::Constant(input.size(), 1);
}
