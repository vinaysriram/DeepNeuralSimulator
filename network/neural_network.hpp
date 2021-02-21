#ifndef NETWORK_H_
#define NETWORK_H_

#include <math.h>

#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <random>

enum class LayerType { Linear, ReLU, Logistic, Tanh, Softmax };

struct LayerOutputs {
  std::map<int, Eigen::VectorXd> a;  // Post-activation layer outputs
  std::map<int, Eigen::VectorXd> z;  // Pre-activation layer outputs
};

struct LayerGradients {
  std::map<int, Eigen::MatrixXd> dW;  // Gradients of the weight matrices
  std::map<int, Eigen::VectorXd> db;  // Gradients of the bias vectors
};

class Network {
 public:
  // Creates a network given two lists. The first, layerTypes, specifies the
  // activation functions of the hidden layers and output layer. The second,
  // layerDimensions, specified the number of neurons in each of the layers.
  // The first entry in layerDimensions is the input dimension, the last is the
  // output dimension, and all other entries are the hidden layer dimensions.
  // Thus, layerDimensions must have a length (L) of at least 2 and layerTypes
  // must have length L - 1 (since no activation is applied on the inputs).
  Network(const std::vector<LayerType> &layerTypes,
          const std::vector<int> &layerDimensions);

  // Trains on the supplied data via stochastic gradient descent.
  void Train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
             double learningRate = 0.01, double regularization = 0.0,
             int epochs = 100);

  // Runs the network on a single sample input.
  Eigen::VectorXd RunSample(const Eigen::VectorXd &x);

  // Runs the network on an entire dataset.
  Eigen::MatrixXd RunDataset(const Eigen::MatrixXd &x);

 private:
  std::map<int, Eigen::MatrixXd> W;   // Weight matrices
  std::map<int, Eigen::VectorXd> b;   // Bias vectors
  std::vector<int> layerDimensions;   // Layer dimensions
  std::vector<LayerType> layerTypes;  // Layer types

  // Serialize and deserialize LayerType.
  LayerType LabelToLayerType(int layerType);
  int LayerTypeToLabel(LayerType layerType);

  // Apply the specified activation function to the raw layer output.
  Eigen::VectorXd Activation(const Eigen::VectorXd &z, LayerType type);

  // Compute the gradient of the specified activation function.
  Eigen::VectorXd Gradient(const Eigen::VectorXd &z, LayerType type);

  // Computes pre/post activation outputs for a single sample.
  LayerOutputs ForwardPropagation(const Eigen::VectorXd &x);

  // Computes gradients for a single training sample pair.
  LayerGradients BackwardPropagation(const Eigen::VectorXd &x,
                                     const Eigen::VectorXd &y);
};

#endif