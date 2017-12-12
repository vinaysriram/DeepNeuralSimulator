
#include "AbstractLayer.hpp"
#include "SpikingLayer.hpp"
#include "SoftmaxLayer.hpp"
#include "SigmoidLayer.hpp"
#include "ReLuLayer.hpp"
#include "TanhLayer.hpp"
#include <iostream>
#include <random>
#include <vector>

using namespace Eigen;
using namespace std;

int main(int argc, char **argv)
{
  /* Inputs and Outputs */
  VectorXd testInput(1);
  VectorXd testOutput(1);

  /* Random Number Engine */
  default_random_engine gen;

  /* Layers */
  vector<AbstractLayer *> layers;
  layers.push_back(new SoftmaxLayer(1, 1, gen));
  layers.push_back(new SigmoidLayer(1, 1, gen));
  layers.push_back(new ReLuLayer(1, 1, gen));
  layers.push_back(new TanhLayer(1, 1, gen));
  layers.push_back(new SpikingLayer(1, 1, gen, 5));
  
  /* Test Activations/Gradients */
  for(double current = -50.0; current < 50.0; current += 0.01) {
    testInput(0) = current;
    for(int i = 0; i < 4; i++) {
      layers[i]->activation(testInput, testOutput);
      cout << testOutput << ", ";
    }
    layers[4]->activation(testInput, testOutput);
    cout << testOutput << endl;
    cerr << current << endl;
  }
  
  /* Test Matrix Functionality */
  
  SigmoidLayer sigLayer(5, 3, gen);
  Eigen::VectorXd input(3);
  input(0) = 1; input(1) = 1; input(2) = 1;
  Eigen::VectorXd output;
  cerr << sigLayer.b << endl;
  cerr << sigLayer.W << endl;
  sigLayer.computeScaled(input, output);
  cerr << output << endl;
  
}
