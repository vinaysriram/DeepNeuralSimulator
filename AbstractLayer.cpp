
#include "AbstractLayer.hpp"
#include <iostream>

using namespace Eigen;
using namespace std;

AbstractLayer::AbstractLayer(int numNeurons, int numInputs, default_random_engine &gen)
{
  inDim = numInputs;
  outDim = numNeurons;
  b = VectorXd(numNeurons);
  W = MatrixXd(numNeurons, numInputs);
  normal_distribution<double> dist(0.0, 1.0);
  for(int i = 0; i < numNeurons; i++) {
    for(int j = 0; j < numInputs; j++) {
      W(i, j) = dist(gen);
    }
    b(i) = dist(gen);
  }
}

AbstractLayer::~AbstractLayer() {}

void AbstractLayer::computeScaled(VectorXd &input, VectorXd &output)
{
  output = (W * input) + b;
}
